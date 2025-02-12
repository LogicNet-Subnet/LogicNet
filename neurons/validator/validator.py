import os
from dotenv import load_dotenv
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
from logicnet.validator import MinerManager, LogicChallenger, LogicRewarder, MinerInfo
from logicnet.utils.wandb_manager import WandbManager
from logicnet.utils.text_uts import modify_question
from logicnet.protocol import LogicSynapse
from neurons.validator.core.serving_queue import QueryQueue
from collections import defaultdict
import wandb


def init_category(config=None, model_rotation_pool=None, dataset_weight=None):
    category = {
        "Logic": {
            "synapse_type": ln.protocol.LogicSynapse,
            "incentive_weight": 1.0,
            "challenger": LogicChallenger(model_rotation_pool, dataset_weight),
            "rewarder": LogicRewarder(model_rotation_pool),
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

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        """
        MAIN VALIDATOR that run the synthetic epoch and opening a proxy for receiving queries from the world.
        """
        super(Validator, self).__init__(config=config)
        bt.logging.info("\033[1;32m🧠 load_state()\033[0m")

        ### Initialize model rotation pool ###
        self.model_rotation_pool = {}
        openai_key = os.getenv("OPENAI_API_KEY")
        togetherai_key = os.getenv("TOGETHERAI_API_KEY")
        if not openai_key and not togetherai_key:
            bt.logging.warning("OPENAI_API_KEY or TOGETHERAI_API_KEY is not set. Please set it to use OpenAI or TogetherAI.")
            raise ValueError("OPENAI_API_KEY or TOGETHERAI_API_KEY is not set. Please set it to use OpenAI or TogetherAI and restart the validator.")
        
        base_urls = self.config.llm_client.base_urls.split(",")
        models = self.config.llm_client.models.split(",")

        # Ensure the lists have enough elements
        # if len(base_urls) < 3 or len(models) < 3:
        #     bt.logging.warning("base_urls or models configuration is incomplete. Please ensure they have just 3 entries.")
        #     raise ValueError("base_urls or models configuration is incomplete. Please ensure they have just 3 entries.")

        if len(base_urls) < 1 or len(models) < 1:
            bt.logging.warning(
                "base_urls or models configuration is incomplete. Please ensure they have at least 1 entry."
            )
            raise ValueError(
                "base_urls or models configuration is incomplete. Please ensure they have at least 1 entry."
            )
        
        self.model_rotation_pool = {
            # "vllm": [base_urls[0].strip(), "xyz", models[0]],
            "openai": [base_urls[1].strip(), openai_key, models[1]],
            # "togetherai": [base_urls[2].strip(), togetherai_key, models[2]],
        }
        # for key, value in self.model_rotation_pool.items():
        #     if value[2] in model_blacklist:
        #         bt.logging.warning(f"Model {value[2]} is blacklisted. Please use another model.")
        #         self.model_rotation_pool[key] = "no use"
        
        # Immediately blacklist if it's not "gpt-4o" and force it to be "gpt-4o"
        if "gpt-4o" not in self.model_rotation_pool["openai"][2]:
            bt.logging.warning(
                f"Model must be gpt-4o. Found {self.model_rotation_pool['openai'][2]} instead."
            )
            bt.logging.info("Setting OpenAI model to gpt-4o.")
            self.model_rotation_pool["openai"][2] = "gpt-4o"
        
        # Check if 'null' is at the same index in both cli lsts
        for i in range(3):
            if base_urls[i].strip() == 'null' or models[i].strip() == 'null':
                if i == 0:
                    self.model_rotation_pool["vllm"] = "no use"
                elif i == 1:
                    self.model_rotation_pool["openai"] = "no use"
                elif i == 2:
                    self.model_rotation_pool["togetherai"] = "no use"
        
        # Check if all models are set to "no use"
        if all(value == "no use" for value in self.model_rotation_pool.values()):
            bt.logging.warning("All models are set to 'no use'. Validator cannot proceed.")
            raise ValueError("All models are set to 'no use'. Please configure at least one model and restart the validator.")
        
        # Create a model_rotation_pool_without_keys
        model_rotation_pool_without_keys = {
            key: "no use" if value == "no use" else [value[0], "Not allowed to see.", value[2]]
            if key in ["openai", "togetherai"] else value
            for key, value in self.model_rotation_pool.items()
        }
        bt.logging.info(f"Model rotation pool without keys: {model_rotation_pool_without_keys}")

        self.categories = init_category(self.config, self.model_rotation_pool, self.config.dataset_weight)
        self.miner_manager = MinerManager(self)
        self.load_state()
        self.update_scores_on_chain()
        self.sync()
        self.miner_manager.update_miners_identity()
        self.wandb_manager = WandbManager(neuron = self)
        self.query_queue = QueryQueue(
            list(self.categories.keys()),
            time_per_loop=self.config.loop_base_time,
        )
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

    def forward(self):
        """
        Query miners by batched from the serving queue then process challenge-generating -> querying -> rewarding in background by threads
        DEFAULT: 16 miners per batch, 600 seconds per loop.
        """
        self.store_miner_infomation()
        bt.logging.info("\033[1;34m🔄 Updating available models & uids\033[0m")
        async_batch_size = self.config.async_batch_size
        loop_base_time = self.config.loop_base_time  # default is 600 seconds
        threads = []
        loop_start = time.time()
        self.miner_manager.update_miners_identity()
        self.query_queue.update_queue(self.miner_manager.all_uids_info)
        self.miner_uids = []
        self.miner_scores = []
        self.miner_reward_logs = []

        # Set up wandb log
        if not self.config.wandb.off:
            today = datetime.date.today()
            if (self.wandb_manager.wandb_start_date != today and 
                hasattr(self.wandb_manager, 'wandb') and 
                self.wandb_manager.wandb is not None):
                self.wandb_manager.wandb.finish()
                self.wandb_manager.init_wandb()

        # Query and reward
        for (
            category,
            uids,
            should_rewards,
            sleep_per_batch,
        ) in self.query_queue.get_batch_query(async_batch_size):
            bt.logging.info(
                f"\033[1;34m🔍 Querying {len(uids)} uids for model {category}, sleep_per_batch: {sleep_per_batch}\033[0m"
            )

            thread = threading.Thread(
                target=self.async_query_and_reward,
                args=(category, uids, should_rewards),
            )
            threads.append(thread)
            thread.start()

            bt.logging.info(
                f"\033[1;34m😴 Sleeping for {sleep_per_batch} seconds between batches\033[0m"
            )
            time.sleep(sleep_per_batch)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Assign incentive rewards
        self.assign_incentive_rewards(self.miner_uids, self.miner_scores, self.miner_reward_logs)

        # Assign incentive rewards
        self.assign_incentive_rewards(
            self.miner_uids,
            self.miner_scores,
            self.miner_reward_logs
        )

        # Flatten logs for passing to wandb
        flat_reward_logs = []
        for batch_logs in self.miner_reward_logs:
            flat_reward_logs.extend(batch_logs)

        # Log to wandb
        if not self.config.wandb.off:
            self._log_wandb(flat_reward_logs)

        # Update scores on chain
        self.update_scores_on_chain()
        self.save_state()
        self.store_miner_infomation()

        actual_time_taken = time.time() - loop_start

        if actual_time_taken < loop_base_time:
            bt.logging.info(
                f"\033[1;34m😴 Sleeping for {loop_base_time - actual_time_taken} seconds\033[0m"
            )
            time.sleep(loop_base_time - actual_time_taken)

    def async_query_and_reward(
        self,
        category: str,
        uids: list[int],
        should_rewards: list[int],
    ):
        dendrite = bt.dendrite(self.wallet)
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

            ## loop for each miner, add noise and send the synapse to the miner
            # responses = []
            # for axon in axons:
            #     noise_synapse = self.add_noise_to_synapse_question(synapse)
            #     response = dendrite.query(
            #         axons=axon,
            #         synapse=noise_synapse,
            #         deserialize=False,
            #         timeout=self.categories[category]["timeout"],
            #     )
            #     responses.append(response)
            responses = dendrite.query(
                axons=axons,
                synapse=synapse,
                deserialize=False,
                timeout=self.categories[category]["timeout"],
            )

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

                for i, uid in enumerate(reward_uids):
                    if rewards[i] > 0:
                        rewards[i] = rewards[i] * (
                            0.9
                            + 0.1 * self.miner_manager.all_uids_info[uid].reward_scale
                        )

                unique_logs = {}
                for log in reward_logs:
                    miner_uid = log["miner_uid"]
                    if miner_uid not in unique_logs:
                        unique_logs[miner_uid] = log

                logs_str = []
                for log in unique_logs.values():
                    logs_str.append(
                        f"Task ID: [{log['task_uid']}], Miner UID: {log['miner_uid']}, Reward: {log['reward']}, Correctness: {log['correctness']}, Similarity: {log['similarity']}, Process Time: {log['process_time']}, Miner Response: {log['miner_response']};"
                    )
                formatted_logs_str = json.dumps(logs_str, indent = 5)
                bt.logging.info(f"\033[1;32m🏆 Miner Scores: {formatted_logs_str}\033[0m")

                if rewards and reward_logs and uids: 
                    self.miner_reward_logs.append(reward_logs)
                    self.miner_uids.append(uids) 
                    self.miner_scores.append(rewards)

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
            rank = i + 1 if score != previous_score else rank
            ranks.append((reward_id, rank, score))
            previous_score = score
        
        # Restore the original order
        ranks.sort(key=lambda x: x[0])

        # Calculate incentive rewards based on the rank, applying the cubic function for positive scores
        def incentive_formula(rank):
            reward_value = -1.038e-7 * rank**3 + 6.214e-5 * rank**2 - 0.0129 * rank - 0.0118
            # Scale up the reward value between 0 and 1
            scaled_reward_value = reward_value + 1
            return scaled_reward_value
        
        incentive_rewards = [
            (incentive_formula(rank) if score > 0 else 0) for _, rank, score in ranks
        ]
        
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
        # The batch size is 8 or the number of miners
        batch_size = min(4, model_miner_count)
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

    def _log_wandb(self, all_reward_logs: list):
        """
        Log reward logs to wandb as a table with the following columns:
            [
                "miner_uid",
                "task_uid_list",
                "miner_response_list",
                "final_score_list",
                "mean_final_score",
                "correctness_list",
                "mean_correctness",
                "similarity_list",
                "mean_similarity",
                "processing_time_list",
                "mean_processing_time",
            ]
        Each row in the table is for one UID, containing lists of their tasks, responses, scores, etc.
        """
        # 1) Guard clauses
        if not self.wandb_manager or not self.wandb_manager.wandb:
            bt.logging.warning("Wandb is not initialized. Skipping logging.")
            return
        if not all_reward_logs:
            bt.logging.warning("No reward logs available. Skipping wandb logging.")
            return

        # 2) Group logs by miner_uid
        logs_by_uid = defaultdict(list)
        for log in all_reward_logs:
            logs_by_uid[log["miner_uid"]].append(log)

        # 3) Prepare a wandb.Table with the requested columns
        table_columns = [
            "miner_uid",
            "task_uid_list",
            "miner_response_list",
            "final_score_list",
            "mean_final_score",
            "correctness_list",
            "mean_correctness",
            "similarity_list",
            "mean_similarity",
            "processing_time_list",
            "mean_processing_time",
        ]
        miner_logs_table = wandb.Table(columns=table_columns)

        # Variables to compute overall means across *all* logs
        all_final_scores = []
        all_correctnesses = []
        all_similarities = []
        all_process_times = []

        # 4) Build each row for each UID
        for uid, logs in logs_by_uid.items():
            # Gather lists
            task_ids = []
            miner_responses = []
            final_scores = []
            correctness_list = []
            similarity_list = []
            process_times = []

            for l in logs:
                task_ids.append(l["task_uid"])
                miner_responses.append(l["miner_response"])
                final_scores.append(l["reward"])  # rename reward -> final_score
                correctness_list.append(l["correctness"])
                similarity_list.append(l["similarity"])
                process_times.append(l["process_time"])

            # Means for this UID
            mean_final_score = sum(final_scores) / len(final_scores)
            mean_correctness = sum(correctness_list) / len(correctness_list)
            mean_similarity = sum(similarity_list) / len(similarity_list)
            mean_process_time = sum(process_times) / len(process_times)

            # Add to overall aggregator
            all_final_scores.extend(final_scores)
            all_correctnesses.extend(correctness_list)
            all_similarities.extend(similarity_list)
            all_process_times.extend(process_times)

            # Add a single table row for this UID
            miner_logs_table.add_data(
                uid,
                task_ids,
                miner_responses,
                final_scores,
                round(mean_final_score, 4),
                correctness_list,
                round(mean_correctness, 4),
                similarity_list,
                round(mean_similarity, 4),
                process_times,
                round(mean_process_time, 4),
            )

        # 5) Compute overall means across all logs
        overall_mean_score = sum(all_final_scores) / len(all_final_scores)
        overall_mean_correctness = sum(all_correctnesses) / len(all_correctnesses)
        overall_mean_similarity = sum(all_similarities) / len(all_similarities)
        overall_mean_process_time = sum(all_process_times) / len(all_process_times)

        # 6) Prepare the final data to log
        wandb_data = {
            "epoch_or_step": self.step,
            "num_logs_total": len(all_reward_logs),
            "overall_mean_score": round(overall_mean_score, 4),
            "overall_mean_correctness": round(overall_mean_correctness, 4),
            "overall_mean_similarity": round(overall_mean_similarity, 4),
            "overall_mean_process_time": round(overall_mean_process_time, 4),
            "miner_logs_table": miner_logs_table,  # The actual table
        }

        # 7) Log to wandb
        self.wandb_manager.wandb.log(wandb_data, commit=True)

        # 8) Debug
        bt.logging.info(
            f"Logged to wandb (epoch_or_step={self.step}):\n"
            f"{json.dumps(wandb_data, indent=2, default=str)}"
        )


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("\033[1;32m🟢 Validator running...\033[0m", time.time())
            time.sleep(360)