import random
from concurrent import futures

import bittensor as bt
import openai
import sympy
import torch
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from logicnet.protocol import LogicSynapse
from logicnet.utils.model_selector import model_selector
from logicnet.utils.regex_helper import extract_numbers
from logicnet.validator.prompt import (CORRECTNESS_TEMPLATE,
                                       DETECT_TRICK_TEMPLATE,
                                       EXTRACT_ANSWER_PROMPT)

SIMILARITY_WEIGHT = 0.3
CORRECTNESS_WEIGHT = 0.7
PROCESSING_TIME_WEIGHT = -0.05



class LogicRewarder:
    def __init__(self, model_rotation_pool: dict, penalty_threshold: float):
        """
        READ HERE TO LEARN HOW VALIDATOR REWARD THE MINER
        """
        self.model_rotation_pool = model_rotation_pool
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.penalty_threshold = penalty_threshold

    def __call__(self, reward_uids, reward_responses: list[LogicSynapse], non_reward_responses: list[LogicSynapse], base_synapse: LogicSynapse):
        """
        Calculate reward for each response using similarity, correctness, and processing time.
        This method also apply a penalty to reponses that have high similarity to other responses in the same batch, which imply they are copying each other.

        Args:
            reward_uids (list[int]): List of miner UIDs to be rewarded.
            reward_responses (list[LogicSynapse]): Synapse responses from miners to be rewarded.
            non_reward_responses (list[LogicSynapse]): Synapse responses from miners to be penalized.
            base_synapse (LogicSynapse): Base synapse containing the ground truth and raw logic question.

        Returns:
            list[float]: List of rewards for each response.
        """
        # Get the unique task UID from the base_synapse
        task_uid = base_synapse.task_uid

        valid_reward_uids = [
            uid for uid, response in zip(reward_uids, reward_responses) if response.is_success
        ]
        valid_reward_responses = [response for response in reward_responses if response.is_success]
        valid_non_reward_responses = [response for response in non_reward_responses if response.is_success]
        invalid_reward_uids = [
            uid for uid, response in zip(reward_uids, reward_responses) if not response.is_success
        ]
        bt.logging.info(f"Valid reward UIDs: {valid_reward_uids}")
        bt.logging.info(f"Invalid reward UIDs: {invalid_reward_uids}")
        invalid_rewards = [0 for _ in invalid_reward_uids]
        reward_logs = []
        valid_rewards = []

        if valid_reward_uids:
            ref_ground_truth: str = self._get_ground_truth(
                base_synapse.raw_logic_question
            )
            reward_response_texts = [response.logic_reasoning for response in valid_reward_responses]
            non_reward_response_texts = [response.logic_reasoning for response in valid_non_reward_responses]

            similarities = self._get_similarity(ref_ground_truth, reward_response_texts)
            correctness = self._get_correctness(base_synapse, valid_reward_responses)
            penalties = self._get_penalties(reward_response_texts, non_reward_response_texts)
            process_times = [
                response.dendrite.process_time for response in valid_reward_responses
            ]
            timeout = base_synapse.timeout

            for i in range(len(valid_reward_responses)):
                reward = (
                    SIMILARITY_WEIGHT * similarities[i]
                    + CORRECTNESS_WEIGHT * correctness[i]
                    + PROCESSING_TIME_WEIGHT * min(process_times[i] / timeout, 1)
                )

                # Scale up the reward
                reward = reward / 2 + 0.5

                # Apply penalty to the reward
                if penalties[i] > self.penalty_threshold:
                    reward = reward * (1 - penalties[i])

                # Scale up the reward
                reward = reward / 2 + 0.5
                valid_rewards.append(reward)

                try:
                    reward_info = {
                        "task_uid": task_uid,
                        "miner_uid": valid_reward_uids[i],
                        "reward": reward,
                        "similarity": similarities[i],
                        "correctness": correctness[i],
                        "process_time": process_times[i],
                        "penalty": penalties[i],
                        "miner_response": valid_reward_responses[i].logic_answer.strip(),
                        "miner_reasoning": reward_response_texts[i],
                        "question": base_synapse.raw_logic_question,
                        "logic_question": base_synapse.logic_question,
                        "ground_truth": base_synapse.ground_truth_answer,
                        "ref_ground_truth": ref_ground_truth,
                    }

                    reward_logs.append(reward_info)               
                except Exception as e:
                    bt.logging.error(f"Error in logging reward for valid miners: {e}")

        total_uids = valid_reward_uids + invalid_reward_uids
        rewards = valid_rewards + invalid_rewards

        for invalid_uid in invalid_reward_uids:
            reward_logs.append({
                "task_uid": task_uid,
                "miner_uid": invalid_uid,
                "reward": 0,
                "similarity": 0,
                "correctness": 0,
                "process_time": 0,
                "penalty": 0,
                "miner_response": "",
                "miner_reasoning": "",
                "question": base_synapse.raw_logic_question,
                "logic_question": base_synapse.logic_question,
                "ground_truth": base_synapse.ground_truth_answer,
                "ref_ground_truth": "",
            })

        return total_uids, rewards, reward_logs

    def _get_correctness(
        self, base_synapse: LogicSynapse, responses: list[LogicSynapse]
    ):
        """Calculate the correctness score for each response.

        Args:
            base_synapse (LogicSynapse): The base synapse containing the ground truth and raw logic question.
            responses (list[LogicSynapse]): List of miner responses.

        Returns:
            list[float]: List of correctness scores for each response (float between 0 and 1).
        """
        model, base_url, api_key = model_selector(self.model_rotation_pool)
        if not model:
            raise ValueError("Model ID is not valid or not provided.")
        if not base_url:
            raise ValueError("Base URL is not valid or not provided.")
        if not api_key:
            raise ValueError("API key is not valid or not provided.")
        
        openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        bt.logging.info(f"Initiating request with model '{model}' at base URL '{base_url}'.")

        ground_truth_answer = base_synapse.ground_truth_answer
        bt.logging.info(f"[CORRECTNESS] Ground truth: {ground_truth_answer}")
        correctness = []
        batch_llm_inputs = []
        indices_for_llm = []

        for idx, response in enumerate(responses):
            miner_answer = response.logic_answer.strip()
            # bt.logging.info(f"[CORRECTNESS] Miner response: {miner_answer}")
            # Try programmatic comparison
            score = self._compare_numerical_answers(ground_truth_answer, miner_answer)
            if score is not None:
                correctness.append(score)
                bt.logging.info(f"[CORRECTNESS] Used programmatic comparison for response {idx} with score {score}")
            else:
                # Need LLM evaluation
                bt.logging.info(f"[CORRECTNESS] Unable to use programmatic comparison. Need LLM evaluation for response")
                correctness.append(0)  # Placeholder
                batch_llm_inputs.append({
                    "question": base_synapse.raw_logic_question,
                    "ground_truth_answer": ground_truth_answer,
                    "response": miner_answer
                })
                # log bt.debug for what score did the LLM give
                indices_for_llm.append(idx)

        if batch_llm_inputs:
            with futures.ThreadPoolExecutor() as executor:
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        llm_scores = executor.map(
                            lambda inputs: self._get_correctness_by_llm(
                                question=inputs["question"],
                                ground_truth=inputs["ground_truth_answer"],
                                response=inputs["response"],
                                model_name=model,
                                openai_client=openai_client,
                            ),
                            batch_llm_inputs,
                        )
                        for idx, score in zip(indices_for_llm, llm_scores):
                            bt.logging.info(f"[CORRECTNESS] LLM Rating: {score}")
                            correctness[idx] = score
                        break
                    except Exception as e:
                        bt.logging.error(f"Error in compute score by llm model: {e}")
                        for idx in indices_for_llm:
                            correctness[idx] = 0.5
        return correctness
    
    def clean_response(self, response: str):
        """Clean the response by removing formatting characters.

        Args:
            response (str): Raw response.

        Returns:
            str: Cleaned response.
        """
        formatting_chars = ['$', '$$', '\\[', '\\]', '%', '-', "<", ">", "/", "*", "#", "!"]
        for char in formatting_chars:
            response = response.replace(char, ' ')
        return response
    

    def _get_correctness_by_llm(self, question: str, ground_truth: str, response: str, model_name: str, openai_client: openai.OpenAI):
        """Calculate the correctness score for a single response using LLM.

        Args:
            question (str): Raw logic question.
            ground_truth (str): Ground truth answer.
            response (str): Miner's answer.
            model_name (str): Model name for the LLM.
            openai_client (openai.OpenAI): OpenAI client for API requests.

        Returns:
            float: Correctness score for the response (float between 0 and 1).
        """
        # response = response.replace("\n---", "").replace("---\n", "")
        if response.strip() == ";":
            return 0.0
        ## check trick case
        try:
            ## check with hard rule
            strings = ['a', 'b', 'c', 'd', 'e'] ## add to response to avoid gpt cached the output
            cheat_words = ["miner_answer", "<example>", "</", "preference>", "<preference"]
            for cheat_word in cheat_words:
                if cheat_word in response.lower():
                    return -1
                
            clone_response = self.clean_response(response)
            clone_response = str(random.choice(strings)) + clone_response + str(random.choice(strings))
            response_str = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": DETECT_TRICK_TEMPLATE.format(
                            response=clone_response
                        ),
                    },
                ],
                max_tokens=25,
                temperature=0,
            ).choices[0].message.content.strip().lower()
            bt.logging.info(f"[CORRECTNESS] Trick detection: {response_str} ====> {response[:100]}")
            if "yes" in response_str:
                return -1
        except Exception as e:
            bt.logging.error(f"API request failed: {e}")
        
        try:
            if len(response.split()) < 20:
                extraced_miner_answer = response
            else:
                extraced_miner_answer = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": EXTRACT_ANSWER_PROMPT.format(
                                response=response,
                                question=question
                            ),
                        },
                    ],
                    max_tokens=25,
                    temperature=0,
                ).choices[0].message.content.strip().lower()
                if "not_found" in extraced_miner_answer or "not found" in extraced_miner_answer:
                    bt.logging.info(f"[CORRECTNESS] Extracted answer not found: {response}")
                    return 0.0
                else:
                    bt.logging.info(f"[CORRECTNESS] Extracted answer: {extraced_miner_answer}")

            response_str = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": CORRECTNESS_TEMPLATE.format(
                            question=question,
                            ground_truth_answer=ground_truth,
                            response=extraced_miner_answer
                        ),
                    },
                ],
                max_tokens=15,
                temperature=0,
            ).choices[0].message.content.strip().lower()
            bt.logging.info(f"[CORRECTNESS] Rating: {response_str}")
            try:
                correctness_score = float(response_str)
                return min(max(correctness_score, 0.0), 1.0)
            except Exception as e:
                bt.logging.warning(f"Failed to parse correctness score. Assigning default score of 0.5.")
                if "1" in response_str:
                    return 1.0
                return 0.5
        except openai.OpenAIError as e:
            bt.logging.error(f"API request failed: {e}")
            return 0.5
        except Exception as e:
            bt.logging.error(f"Error in compute score by llm model: {e}")
            return 0.5

    def _compare_numerical_answers(self, ground_truth: str, miner_answer: str):
        try:
            # Remove formatting characters from the answers
            formatting_chars = ['$', '$$', '\\[', '\\]', '%']
            for char in formatting_chars:
                ground_truth = ground_truth.replace(char, '')
                miner_answer = miner_answer.replace(char, '')

            # Extract numerical values
            gt_values = extract_numbers(ground_truth)
            miner_values = extract_numbers(miner_answer)

            if len(gt_values) == 0:
                return None

            if len(gt_values) > 0 and len(miner_values) == 0:
                return 0.0

            if len(gt_values) == 1 and len(miner_values) == 1:
                # Single numerical value found in both answers
                gt_value = gt_values[0]
                miner_value = miner_values[0]
                abs_difference = abs(gt_value - miner_value)
                epsilon = 1e-8
                gt_abs = abs(gt_value) + epsilon
                relative_error = abs_difference / gt_abs
                # Logs for debugging
                bt.logging.info(f"[CORRECTNESS DEBUG FOR NUMERICAL COMPARISON]: Absolute difference: {abs_difference}, Relative error: {relative_error}")
                correctness_score = max(0.0, 1.0 - relative_error)
                correctness_score = min(correctness_score, 1.0)
                return correctness_score
            return None
        except Exception as e:
            # Log the problematic input for debugging
            bt.logging.warning(
                f"Failed to _compare_numerical_answers. Error: {e}"
            )
            # Return None so that LLM-based correctness check will be used.
            return None

    def _get_similarity(self, ground_truth: str, responses: list[str]):
        """Calculate cosine similarity between self-generated ground truth and miner responses.

        Args:
            ground_truth (str): Ground truth generated by self.
            responses (list[str]): List of responses from miners.

        Returns:
            list[float]: List of similarity scores for each response.
        """
        try:
            ground_truth_embedding = self.embedder.encode(ground_truth, convert_to_tensor=True).cpu()
            response_embeddings = self.embedder.encode(responses, convert_to_tensor=True).cpu()

            # Calculate similarity
            similarities = torch.nn.functional.cosine_similarity(
                ground_truth_embedding,
                response_embeddings,
                dim=1,
            )
            return similarities.tolist()
        except Exception as e:
            bt.logging.warning(f"Failed to calculate similarity.\nError: {e}")
            return [0.5] * len(responses)
        
    def _get_penalties(self, reward_responses_texts: list[str], non_reward_responses_texts: list[str]):
        """Calculate penalties for each response.
        Penalty is calculated based on the similarity between the response and the other responses using embedding similarity and Levenshtein distance.
        
        Args:
            reward_responses_texts (list[str]): List of responses from miners to be rewarded.
            non_reward_responses_texts (list[str]): List of responses from miners to be penalized.

        Returns:
            list[float]: List of penalties for each response, 0 if the response is not penalized, 1 if the response is highly penalized.
        """
        penalties = []
        try:
            all_responses = reward_responses_texts + non_reward_responses_texts
            
            # Calculate embedding similarity
            embeddings = self.embedder.encode(all_responses, convert_to_tensor=True).cpu()
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embedding_similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
            # We don't want to penalize the response with itself
            embedding_similarity_matrix = embedding_similarity_matrix - torch.eye(embedding_similarity_matrix.shape[0])
            reward_responses_embedding_similarity = embedding_similarity_matrix[:len(reward_responses_texts)].max(dim=1).values # consider the max similarity for each reward response
            reward_responses_embedding_similarity = (reward_responses_embedding_similarity + 1) / 2

            # Calculate Levenshtein distance
            levenshtein_similarity_matrix = torch.zeros((len(all_responses), len(all_responses)))
            for i in range(len(all_responses)):
                for j in range(len(all_responses)):
                    if i != j:
                        levenshtein_similarity_matrix[i, j] = fuzz.partial_ratio(all_responses[i], all_responses[j]) / 100
                    else:
                        levenshtein_similarity_matrix[i, j] = 0

            reward_responses_levenshtein_similarity = levenshtein_similarity_matrix[:len(reward_responses_texts)].max(dim=1).values # consider the max similarity for each reward response
            
            penalties = (reward_responses_embedding_similarity + reward_responses_levenshtein_similarity) / 2
            penalties = penalties.clip(0, 1)
            penalties = [float(penalty) for penalty in penalties.tolist()]
            return penalties

        except Exception as e:
            bt.logging.warning(f"Failed to calculate penalties.\nError: {e}")
            return [0.0] * len(reward_responses_texts)

    def _get_ground_truth(self, question: str):
        """Generate self-generated ground truth based on the question.

        Args:
            question (str): Raw logic question.

        Returns:
            str: Self-generated ground truth.
        """
        messages = [
            {"role": "user", "content": question},
        ]
        model, base_url, api_key = model_selector(self.model_rotation_pool)
        if not model:
            raise ValueError("Model ID is not valid or not provided.")
        if not base_url:
            raise ValueError("Base URL is not valid or not provided.")
        if not api_key:
            raise ValueError("API key is not valid or not provided.")

        openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        bt.logging.info(f"Initiating request with model '{model}' at base URL '{base_url}'.")

        response = ""
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7,
                )
                response = response.choices[0].message.content
                # bt.logging.info(f"[SIMILARITY] Self-generated ground truth: {response}")
                return response  # Return response if successful
            
            except openai.OpenAIError as e:
                bt.logging.error(f"API request failed on attempt {attempt + 1}: {e}")
                if attempt == 2:  # Last attempt
                    # Switch to another model, base URL, and API key
                    model, base_url, api_key = model_selector(self.model_rotation_pool)
                    if not model or not base_url or not api_key:
                        bt.logging.error("No alternative model, base URL, or API key available.")

                    else:
                        openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
                        bt.logging.info(f"Initiating request with model '{model}' at base URL '{base_url}'.")
                        try:
                            response = openai_client.chat.completions.create(
                                model=model,
                                messages=messages,
                                max_tokens=1024,
                                temperature=0.7,
                            )
                            response = response.choices[0].message.content
                            # bt.logging.info(f"[SIMILARITY] Self-generated ground truth: {response}")
                            return response
                        except openai.OpenAIError as e:
                            bt.logging.error(f"API request failed after switching: {e}")

        return response