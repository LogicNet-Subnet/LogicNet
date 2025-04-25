import random
import math
import bittensor as bt
from logicnet.utils.volume_setting import (
    MIN_RATE_LIMIT,
    MAX_RATE_LIMIT,
)

class QueryItem:
    def __init__(self, uid: int):
        self.uid = uid


class QueryQueue:
    """
    QueryQueue is a list-based storage for the uids for the synthetic and proxy model.
    Created based on the rate limit of miners.
    """

    def __init__(self):
        self.synthentic_queue = []
        self.proxy_queue = []
        self.synthentic_rewarded = {}
        self.current_synthetic_index = 0
        self.current_proxy_index = 0

    def update_queue(self, all_uids_info):
        self.synthentic_rewarded = {}
        self.synthentic_queue = []
        self.proxy_queue = []
        self.current_synthetic_index = 0
        self.current_proxy_index = 0

        all_uids = []

        valid_uids = []
        for uid, info in all_uids_info.items():
            if not info.category:
                continue
            valid_uids.append(uid)
            synthetic_rate_limit, proxy_rate_limit = self.get_rate_limit_by_type(info.rate_limit)
            all_uids.append(QueryItem(uid=uid))

            for _ in range(int(synthetic_rate_limit)):
                self.synthentic_queue.append(QueryItem(uid=uid))
            for _ in range(int(synthetic_rate_limit)):
                self.proxy_queue.append(QueryItem(uid=uid))

        bt.logging.info(f"Valid uids: {valid_uids}")
        random.shuffle(self.synthentic_queue)
        random.shuffle(self.proxy_queue)

    def get_batch_query(self, batch_size: int, batch_number: int):
        """
        Return N batch of query.
        
        Args:
            batch_size (int): Number of queries per batch
            N (int): Number of batches to return

        Returns:
            list: List of N batches of query
        """
        for _ in range(batch_number):
            ## random select batch_size from self.synthentic_queue
            batch_items = [random.choice(self.synthentic_queue) for _ in range(batch_size)]
            uids_to_query = [item.uid for item in batch_items]
            should_rewards = [self.random_should_reward(item.uid) for item in batch_items]
            for uid in uids_to_query:
                if uid not in self.synthentic_rewarded:
                    self.synthentic_rewarded[uid] = 0
                self.synthentic_rewarded[uid] += 1
            yield uids_to_query, should_rewards

    def random_should_reward(self, uid):
        return random.random() < 0.3  # 30% chance of validating and re-computing the reward

    def get_query_for_proxy(self):
        # First yield all synthetic items
        while self.current_synthetic_index < len(self.synthentic_queue):
            query_item = self.synthentic_queue[self.current_synthetic_index]
            self.current_synthetic_index += 1
            should_reward = False
            if (query_item.uid not in self.synthentic_rewarded) or (self.synthentic_rewarded[query_item.uid] <= 20):
                should_reward = True
            yield query_item.uid, should_reward

        # Then yield all proxy items
        while self.current_proxy_index < len(self.proxy_queue):
            query_item = self.proxy_queue[self.current_proxy_index]
            self.current_proxy_index += 1
            yield query_item.uid, False

    def get_rate_limit_by_type(self, rate_limit):
        synthentic_rate_limit = max(1, int(math.floor(rate_limit * 0.8)) - 1)
        synthentic_rate_limit = max(
            rate_limit - synthentic_rate_limit, synthentic_rate_limit
        )
        proxy_rate_limit = rate_limit - synthentic_rate_limit
        return synthentic_rate_limit, proxy_rate_limit