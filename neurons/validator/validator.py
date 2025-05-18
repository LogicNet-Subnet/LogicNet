import os
from dotenv import load_dotenv
import asyncio
load_dotenv()
import pickle
import time
import json
import re
import threading
import datetime
import random
import traceback
import torch
import requests
from copy import deepcopy
import bittensor as bt
import logicnet as ln
from neurons.validator.validator_proxy import ValidatorProxy
from logicnet.base.validator import BaseValidatorNeuron
from logicnet.validator import MinerManager, LogicChallenger, LogicRewarder
from logicnet.utils.text_uts import modify_question
from logicnet.protocol import LogicSynapse
from neurons.validator.core.serving_queue import QueryQueue
from threading import Lock
import queue
from logicnet.utils.minio_manager import MinioManager
import glob

log_bucket_name = "logs"
app_name = os.getenv("APP_NAME", "sn35-validator")
validator_username = os.getenv("VALIDATOR_USERNAME")
minio_endpoint = os.getenv("MINIO_ENDPOINT")
access_key = os.getenv("MINIO_ACCESS_KEY")
secret_key = os.getenv("MINIO_SECRET_KEY")
pm2_log_dir = os.getenv("PM2_LOG_DIR", "/root/.pm2/logs")
last_err_file_name = ""
last_out_file_name = ""

# check if the pm2_log_dir is valid
if not os.path.exists(pm2_log_dir):
    bt.logging.error(f"PM2 log directory does not exist: {pm2_log_dir}")
    raise ValueError(f"PM2 log directory does not exist: {pm2_log_dir}")

bt.logging.info(f"PM2_LOG_DIR: {pm2_log_dir}")
bt.logging.info(f"APP_NAME: {app_name}")
bt.logging.info(f"VALIDATOR_USERNAME: {validator_username}")
bt.logging.info(f"MINIO_ENDPOINT: {minio_endpoint}")

def init_category(config=None, model_pool=None):
    category = {
        "Logic": {
            "synapse_type": ln.protocol.LogicSynapse,
            "incentive_weight": 1.0,
            "challenger": LogicChallenger(model_pool),
            "rewarder": LogicRewarder(model_pool),
            "timeout": 64,
        }
    }
    return category


## low quality models
model_blacklist = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct"
]

def get_latest_previous_log_file(log_files):
    """Return the second-most-recent log file based on modification time."""
    if len(log_files) < 2:
        return None  # Not enough files to have a "previous" file
    # Sort files by modification time (most recent first)
    sorted_files = sorted(log_files, key=lambda x: os.path.getmtime(x), reverse=True)
    return sorted_files[1]  # Second file is the latest previous

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        """
        MAIN VALIDATOR that run the synthetic epoch and opening a proxy for receiving queries from the world.
        """
        super(Validator, self).__init__(config=config)
        bt.logging.info("\033[1;32m🧠 load_state()\033[0m")

        try:
            self.minio_manager = MinioManager(minio_endpoint, access_key, secret_key)
        except Exception as e:
            bt.logging.error(f"Error initializing MinioManager: {e}")

        ### Initialize model rotation pool ###
        self.model_pool = {}
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            bt.logging.warning("OPENAI_API_KEY is not set. Please set it to use OpenAI")
            raise ValueError("OPENAI_API_KEY is not set. Please set it to use OpenAI or restart the validator.")
        
        if self.config.llm_client.gpt_url and self.config.llm_client.gpt_model:
            self.model_pool["openai"] = [
                self.config.llm_client.gpt_url,
                openai_key,
                self.config.llm_client.gpt_model
            ]
        if self.config.llm_client.vllm_url and self.config.llm_client.vllm_model:
            self.model_pool["vllm"] = [
                self.config.llm_client.vllm_url, 
                self.config.llm_client.vllm_key, 
                self.config.llm_client.vllm_model
            ]

        for key, value in self.model_pool.items():
            if value[2] in model_blacklist:
                bt.logging.warning(f"Model {value[2]} is blacklisted. Please use another model.")
                del self.model_pool[key]
        
        # Check if all models are invalid
        if not self.model_pool:
            bt.logging.warning("All models are invalid. Validator cannot proceed.")
            raise ValueError("All models are invalid. Please configure at least one model and restart the validator.")
        
        self.push_logs_to_minio()
        self.categories = init_category(self.config, self.model_pool)
        self.miner_manager = MinerManager(self)
        self.load_state()
        # self.update_scores_on_chain()
        # self.sync()
        # self.miner_manager.update_miners_identity()
        self.query_queue = QueryQueue()
        if self.config.proxy.port:
            try:
                self.validator_proxy = ValidatorProxy(self)
                bt.logging.info(
                    "\033[1;32m🟢 Validator proxy started successfully\033[0m"
                )
            except Exception:
                bt.logging.warning(
                    "\033[1;33m⚠️ Warning, proxy did not start correctly, so no one can query through your validator. "
                    "This means you won't participate in queries from apps powered by this subnet. Error message: "
                    + traceback.format_exc()
                    + "\033[0m"
                )
        self.reward_lock = Lock()
        self.reward_queue = queue.Queue()

    def forward(self):
        """
        Query miners by batched from the serving queue then process challenge-generating -> querying -> rewarding in background by threads
        DEFAULT: 16 miners per batch, 600 seconds per loop.
        """
        # self.store_miner_infomation()
        self.push_logs_to_minio()
        bt.logging.info("\033[1;34m🔄 Updating available models & uids\033[0m")
        loop_base_time = self.config.loop_base_time  # default is 600s
        self.miner_manager.update_miners_identity()
        self.query_queue.update_queue(self.miner_manager.all_uids_info)
        self.miner_uids = []
        self.miner_scores = []
        self.miner_reward_logs = []

        # run in 600s
        loop_start = time.time()
        while time.time() - loop_start < loop_base_time:
            iter_start = time.time()
            threads = []
            for (uids, should_rewards) in self.query_queue.get_batch_query(
                batch_size=self.config.batch_size, 
                batch_number=self.config.batch_number
            ):
                bt.logging.info(
                    f"\033[1;34m🔍 Querying {len(uids)} uids for model {self.config.llm_client.gpt_model}\033[0m"
                )
                thread = threading.Thread(
                    target=self.run_async_query,
                    args=("Logic", uids, should_rewards),
                )
                threads.append(thread)
                thread.start()
                time.sleep(4)

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Process all queued results safely
            with self.reward_lock:
                while not self.reward_queue.empty():
                    bt.logging.info(f"\033[1;32m🟢 Update reward logs for miner {uids}")
                    reward_logs, uids, rewards = self.reward_queue.get()
                    self.miner_reward_logs.append(reward_logs)
                    self.miner_uids.append(uids)
                    self.miner_scores.append(rewards)

            bt.logging.info(f"\033[1;32m🟢 Validator iteration completed in {time.time() - iter_start} seconds\033[0m")
        
        # Assign incentive rewards
        bt.logging.info(f"\033[1;32m🟢 Assign incentive rewards for miner {self.miner_uids}")
        self.assign_incentive_rewards(self.miner_uids, self.miner_scores, self.miner_reward_logs)

        # Update scores on chain
        self.update_scores_on_chain()
        self.save_state()
        # self.store_miner_infomation()
        bt.logging.info(f"\033[1;32m🟢 Validator loop completed in {time.time() - loop_start} seconds\033[0m")


    def push_logs_to_minio(self):
        #########################################################
        # UPLOAD OUT LOG FILES
        global last_err_file_name
        global last_out_file_name

        try:
            bt.logging.info(f"\033[1;32m🟢 Pushing out log files to MinIO\033[0m")
            log_regex = os.path.join(pm2_log_dir, f"*{app_name}*out*.log")
            out_log_files = glob.glob(log_regex)
            bt.logging.info(f"\033[1;32m🟢 Out log files: {out_log_files}, regex: {log_regex}\033[0m")

            current_file_count = len(out_log_files)
            # Detect rotation (new file added)
            if current_file_count >= 2:
                # A new file was created, so upload the latest previous file
                previous_file = get_latest_previous_log_file(out_log_files)
                if previous_file != last_out_file_name and previous_file:
                    last_out_file_name = previous_file
                    file_name = os.path.basename(previous_file)
                    if file_name not in self.minio_manager.get_uploaded_files(log_bucket_name):
                        bt.logging.info(f"Uploading {previous_file} to MinIO")
                        if self.minio_manager.upload_file(previous_file, log_bucket_name, validator_username):
                            bt.logging.info(f"\033[1;32m✅ Uploaded {file_name} to MinIO\033[0m")
            #########################################################


            #########################################################
            # UPLOAD ERR LOG FILES
            err_log_files = glob.glob(os.path.join(pm2_log_dir, f"*{app_name}-error*.log"))
            # bt.logging.info(err_log_files)
            current_file_count = len(err_log_files)

            # Detect rotation (new file added)
            if current_file_count >= 2:
                # A new file was created, so upload the latest previous file
                previous_file = get_latest_previous_log_file(err_log_files)
                if previous_file != last_err_file_name and previous_file:
                    last_err_file_name = previous_file
                    file_name = os.path.basename(previous_file)
                    if file_name not in self.minio_manager.get_uploaded_files(log_bucket_name):
                        bt.logging.info(f"Uploading {previous_file} to MinIO")
                        if self.minio_manager.upload_file(previous_file, log_bucket_name, validator_username):
                            bt.logging.info(f"\033[1;32m✅ Uploaded {file_name} to MinIO\033[0m")
            #########################################################
        except Exception as e:
            bt.logging.error(f"Error uploading log files: {e}")


    def run_async_query(self, category: str, uids: list[int], should_rewards: list[int]):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self.async_query_and_reward(category, uids, should_rewards)
            )
        finally:
            loop.close()

    async def async_query_and_reward(
        self,
        category: str,
        uids: list[int],
        should_rewards: list[int],
    ):
        try:
            dendrite = bt.dendrite(self.wallet)
            try:
                uids_should_rewards = list(zip(uids, should_rewards))
                synapses, batched_uids_should_rewards = self.prepare_challenge(
                    uids_should_rewards, category
                )
                
                for synapse, uids_should_rewards in zip(synapses, batched_uids_should_rewards):
                    uids, should_rewards = zip(*uids_should_rewards)
                    if not synapse:
                        continue
                    base_synapse = synapse.model_copy()
                    synapse = synapse.miner_synapse()
                    bt.logging.info(f"\033[1;34m🧠 Synapse to be sent to miners: {synapse}\033[0m")
                    axons = [self.metagraph.axons[int(uid)] for uid in uids]
                    sent_time = time.time()
                    # Use aquery instead of query
                    responses = await dendrite.aquery(
                        axons=axons,
                        synapse=synapse,
                        deserialize=False,
                        timeout=self.categories[category]["timeout"],
                    )
                    for axon, response in zip(axons, responses):
                        bt.logging.info(f"\033[1;34m🧠 {time.time() - sent_time}s Response from {axon}: {response}\033[0m ")

                    reward_responses = [
                        response
                        for response, should_reward in zip(responses, should_rewards)
                        if should_reward
                    ]
                    reward_uids = [
                        uid for uid, should_reward in zip(uids, should_rewards) if should_reward
                    ]

                    if reward_uids:
                        uids, rewards, reward_logs = self.categories[category]["rewarder"](
                            reward_uids, reward_responses, base_synapse
                        )

                        for i, uid in enumerate(uids):
                            if rewards[i] > 0:
                                rewards[i] = rewards[i] * (
                                    0.9 + 0.1 * self.miner_manager.all_uids_info[uid].reward_scale
                                )

                        unique_logs = {}
                        for log in reward_logs:
                            miner_uid = log["miner_uid"]
                            if miner_uid not in unique_logs:
                                unique_logs[miner_uid] = log

                        logs_str = []
                        for log in unique_logs.values():
                            logs_str.append(
                                f"Task ID: [{log['task_uid']}], Miner UID: {log['miner_uid']}, Reward: {log['reward']}, Correctness: {log['correctness']}, Similarity: {log['similarity']}, Process Time: {log['process_time']}, Miner Response: {log['miner_response']}, Ground Truth: {log['ground_truth']}"
                            )
                        formatted_logs_str = json.dumps(logs_str, indent=5)
                        bt.logging.info(f"\033[1;32m🏆 Miner Scores: {formatted_logs_str}\033[0m")
                        if rewards and reward_logs and uids:
                            # Queue the results instead of directly appending
                            self.reward_queue.put((reward_logs, uids, rewards))

            finally:
                # Ensure dendrite cleanup
                await dendrite.aclose_session()

        except Exception as e:
            bt.logging.error(f"Error in async_query_and_reward: {str(e)}")
            bt.logging.debug(traceback.format_exc())

    def add_noise_to_synapse_question(self, synapse: ln.protocol.LogicSynapse):
        """
        Add noise to the synapse question.
        """
        ##copy the synapse
        copy_synapse = deepcopy(synapse)
        ##modify the question
        copy_synapse.logic_question = modify_question(copy_synapse.logic_question)
        return copy_synapse

    def assign_incentive_rewards(self, uids, rewards, reward_logs):
        """
        Calculate incentive rewards based on the rank.
        Get the incentive rewards for the valid responses using the cubic function and valid_rewards rank.
        """
        # Flatten the nested lists
        flat_uids = [uid for uid_list in uids for uid in uid_list]
        flat_rewards = [reward for reward_list in rewards for reward in reward_list]
        flat_reward_logs = [log for log_list in reward_logs for log in log_list]

        # Create a dictionary to track the all scores per UID
        uids_scores = {}
        uids_logs = {}
        for uid, reward, log in zip(flat_uids, flat_rewards, flat_reward_logs):
            if uid not in uids_scores:
                uids_scores[uid] = []
                uids_logs[uid] = []
            uids_scores[uid].append(reward)
            uids_logs[uid].append(log)

        # Now uids_scores holds all rewards each UID achieved this epoch
        # Convert them into lists for processing
        final_uids = list(uids_scores.keys())
        representative_logs = [logs[0] for logs in uids_logs.values()] 
               
        ## compute mean value of rewards
        final_rewards = [sum(uid_rewards) / len(uid_rewards) for uid_rewards in uids_scores.values()]
        ## set the rewards to 0 if the mean is negative
        final_rewards = [reward if reward > 0 else 0 for reward in final_rewards]

        # Now proceed with the incentive rewards calculation on these mean attempts
        original_rewards = list(enumerate(final_rewards))
        # Sort and rank as before, but now we're dealing with mean attempts.
        
        # Sort rewards in descending order based on the score
        sorted_rewards = sorted(original_rewards, key=lambda x: x[1], reverse=True)
        
        # Calculate ranks, handling ties
        ranks = []
        previous_score = None
        rank = 0
        for i, (reward_id, score) in enumerate(sorted_rewards):
            # rank = i + 1 if score != previous_score else rank
            rank = i + 1
            ranks.append((reward_id, rank, score))
            # previous_score = score
        
        # Restore the original order
        ranks.sort(key=lambda x: x[0])

        # Calculate incentive rewards based on the rank, applying the cubic function for positive scores
        def incentive_formula(rank):
            reward_value = -1.038e-7 * rank**3 + 6.214e-5 * rank**2 - 0.0129 * rank - 0.0118
            # Scale up the reward value between 0 and 1
            scaled_reward_value = reward_value + 1
            return scaled_reward_value
        
        # incentive_rewards = [
        #     (incentive_formula(rank) if score > 0.3 else 0) for _, rank, score in ranks
        # ]

        incentive_rewards = []
        for _, rank, score in ranks:
            ## only give reward top 160 miners, set 0 reward for 90 bad miners
            if score > 0.3 and rank <= 160:
                incentive_rewards.append(incentive_formula(rank))
            else:
                incentive_rewards.append(incentive_formula(250)) # add smallest reward for top 90 bad miners

        bt.logging.info(f"\033[1;32m🟢 Final Uids: {final_uids}\033[0m")
        bt.logging.info(f"\033[1;32m🟢 Incentive rewards: {incentive_rewards}\033[0m")
        self.miner_manager.update_scores(final_uids, incentive_rewards, representative_logs)
        
        # Reset logs for next epoch
        self.miner_scores = []
        self.miner_reward_logs = []
        self.miner_uids = []

    def prepare_challenge(self, uids_should_rewards, category):
        """
        Prepare the challenge for the miners. Continue batching to smaller.
        """
        synapse_type = self.categories[category]["synapse_type"]
        challenger = self.categories[category]["challenger"]
        timeout = self.categories[category]["timeout"]
        model_miner_count = len(
            [
                uid
                for uid, info in self.miner_manager.all_uids_info.items()
                if info.category == category
            ]
        )
        # Return empty synapses if no miners are available
        if model_miner_count == 0:
            print("No miners available")
            return [], []
        # The batch size is 8 or the number of miners
        batch_size = max(min(8, model_miner_count), 1)
        random.shuffle(uids_should_rewards)
        batched_uids_should_rewards = [
            uids_should_rewards[i * batch_size : (i + 1) * batch_size]
            for i in range((len(uids_should_rewards) + batch_size - 1) // batch_size)
        ]
        num_batch = len(batched_uids_should_rewards)

        synapses = []
        for i in range(num_batch):
            synapse = synapse_type(category=category, timeout=timeout)
            synapse = challenger(synapse)
            synapses.append(synapse)
        return synapses, batched_uids_should_rewards

    def update_scores_on_chain(self):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        weights = torch.zeros(len(self.miner_manager.all_uids))
        for category in self.categories.keys():
            model_specific_weights = self.miner_manager.get_model_specific_weights(
                category
            )
            model_specific_weights = (
                model_specific_weights * self.categories[category]["incentive_weight"]
            )
            bt.logging.info(
                f"\033[1;34m⚖️ model_specific_weights for {category}\n{model_specific_weights}\033[0m"
            )
            weights = weights + model_specific_weights

        # Check if rewards contains NaN values.
        if torch.isnan(weights).any():
            bt.logging.warning(
                f"\033[1;33m⚠️ NaN values detected in weights: {weights}\033[0m"
            )
            # Replace any NaN values in rewards with 0.
            weights = torch.nan_to_num(weights, 0)
        self.scores: torch.FloatTensor = weights
        bt.logging.success(f"\033[1;32m✅ Updated scores: {self.scores}\033[0m")

    def save_state(self):
        """Saves the state of the validator to a file using pickle."""
        state = {
            "step": self.step,
            "all_uids_info": self.miner_manager.all_uids_info,
        }
        try:
            # Open the file in write-binary mode
            with open(self.config.neuron.full_path + "/state.pkl", "wb") as f:
                pickle.dump(state, f)
            bt.logging.info("State successfully saved to state.pkl")
        except Exception as e:
            bt.logging.error(f"Failed to save state: {e}")

    def load_state(self):
        """Loads state of  validator from a file, with fallback to .pt if .pkl is not found."""
        # TODO: After a transition period, remove support for the old .pt format.
        try:
            path_pt = self.config.neuron.full_path + "/state.pt"
            path_pkl = self.config.neuron.full_path + "/state.pkl"

            # Try to load the newer .pkl format first
            try:
                bt.logging.info(f"Loading validator state from: {path_pkl}")
                with open(path_pkl, "rb") as f:
                    state = pickle.load(f)

                # Restore state from pickle file
                self.step = state["step"]
                self.miner_manager.all_uids_info = state["all_uids_info"]
                bt.logging.info("Successfully loaded state from .pkl file")
                return  # Exit after successful load from .pkl

            except Exception as e:
                bt.logging.warning(f"Failed to load from .pkl format: {e}")

            # If .pkl loading fails, try to load from the old .pt file (PyTorch format)
            try:
                bt.logging.info(f"Loading validator state from: {path_pt}")
                state = torch.load(path_pt)

                # Restore state from .pt file
                self.step = state["step"]
                self.miner_manager.all_uids_info = state["all_uids_info"]
                bt.logging.info("Successfully loaded state from .pt file")

            except Exception as e:
                bt.logging.error(f"Failed to load from .pt format: {e}")
                self.step = 0  # Default fallback when both load attempts fail
                bt.logging.error("Could not find previously saved state or error loading it.")

        except Exception as e:
            self.step = 0  # Default fallback in case of an unknown error
            bt.logging.error(f"Error loading state: {e}")


    def store_miner_infomation(self):
        miner_informations = self.miner_manager.to_dict()

        def _post_miner_informations(miner_informations):
            # Convert miner_informations to a JSON-serializable format
            serializable_miner_informations = convert_to_serializable(miner_informations)
            
            try:
                response = requests.post(
                    url=self.config.storage.storage_url,
                    json={
                        "miner_information": serializable_miner_informations,
                        "validator_uid": int(self.uid),
                    },
                )
                if response.status_code == 200:
                    bt.logging.info("\033[1;32m✅ Miner information successfully stored.\033[0m")
                else:
                    bt.logging.warning(
                        f"\033[1;33m⚠️ Failed to store miner information, status code: {response.status_code}\033[0m"
                    )
            except requests.exceptions.RequestException as e:
                bt.logging.error(f"\033[1;31m❌ Error storing miner information: {e}\033[0m")

        def convert_to_serializable(data):
            # Implement conversion logic for serialization
            if isinstance(data, dict):
                return {key: convert_to_serializable(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [convert_to_serializable(element) for element in data]
            elif isinstance(data, (int, str, bool, float)):
                return data
            elif hasattr(data, '__float__'):
                return float(data)
            else:
                return str(data)

        thread = threading.Thread(
            target=_post_miner_informations,
            args=(miner_informations,),
        )
        thread.start()

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":        
    with Validator() as validator:
        while True:
            bt.logging.info("\033[1;32m🟢 Validator running...\033[0m", time.time())
            time.sleep(60)