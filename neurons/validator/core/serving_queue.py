import queue
import random
import math
import bittensor as bt


class QueryItem:
    def __init__(self, uid: int):
        self.uid = uid


class QueryQueue:
    """
    QueryQueue is a queue for storing the uids for the synthetic and proxy model.
    Created based on the rate limit of miners.
    """

    def __init__(self, categories: list[str], time_per_loop: int = 600):
        self.synthentic_queue: dict[str, queue.Queue[QueryItem]] = {
            category: queue.Queue() for category in categories
        }
        self.proxy_queue: dict[str, queue.Queue[QueryItem]] = {
            category: queue.Queue() for category in categories
        }
        self.synthentic_rewarded = {}
        self.time_per_loop = time_per_loop
        self.total_uids_remaining = 0

    def update_queue(self, all_uids_info):
        self.total_uids_remaining = 0
        self.synthentic_rewarded = {}
        for q in self.synthentic_queue.values():
            q.queue.clear()
        for q in self.proxy_queue.values():
            q.queue.clear()

        all_uids_by_category = {category: [] for category in self.synthentic_queue}

        for uid, info in all_uids_info.items():
            if not info.category:
                continue
            synthentic_model_queue = self.synthentic_queue.setdefault(
                info.category, queue.Queue()
            )
            proxy_model_queue = self.proxy_queue.setdefault(
                info.category, queue.Queue()
            )
            synthetic_rate_limit, proxy_rate_limit = self.get_rate_limit_by_type(
                info.rate_limit
            )
            if info.category not in all_uids_by_category:
                all_uids_by_category[info.category] = []
            all_uids_by_category[info.category].append(QueryItem(uid=uid))

            for _ in range(int(synthetic_rate_limit)):
                synthentic_model_queue.put(QueryItem(uid=uid))
            for _ in range(int(proxy_rate_limit)):
                proxy_model_queue.put(QueryItem(uid=uid))
        
        # Shuffle the queue
        for category, q in self.synthentic_queue.items():
            shuffled_items = list(q.queue)
            random.shuffle(shuffled_items)
            q.queue.clear()

            # add full list UID at the start of the queue, make sure that all UID is queried at least twice in the loop begining
            for _ in range(2):
                for item in all_uids_by_category[category]:
                    q.put(item)
            
            # add shuffled items to the queue
            for item in shuffled_items:
                q.put(item)

            self.total_uids_remaining += len(q.queue)
            bt.logging.info(
                f"- Model {category} has {len(q.queue)} uids remaining for synthentic"
            )
        for category, q in self.proxy_queue.items():
            random.shuffle(q.queue)
            bt.logging.info(
                f"- Model {category} has {len(q.queue)} uids remaining for organic"
            )

    def get_batch_query(self, batch_size: int):
        if not self.total_uids_remaining:
            return
        more_data = True
        initial_total = self.total_uids_remaining 

        while more_data:
            more_data = False
            # Get non-empty categories
            available_categories = [
                category for category, q in self.synthentic_queue.items() 
                if q and not q.empty()
            ]
            
            if not available_categories:
                continue

            # Randomly select one category
            category = random.choice(available_categories)
            q = self.synthentic_queue.get(category)

            # Calculate progress through queue (0.0 to 1.0)
            progress = 1 - (self.total_uids_remaining / max(initial_total, 1))
            time_to_sleep = self.time_per_loop * min(batch_size / (initial_total + 1), 1) * math.exp(progress) 

            uids_to_query = []
            should_rewards = []

            while len(uids_to_query) < batch_size and not q.empty():
                more_data = True
                query_item = q.get()
                uids_to_query.append(query_item.uid)
                should_rewards.append(self.random_should_reward(query_item.uid))

                if query_item.uid not in self.synthentic_rewarded:
                    self.synthentic_rewarded[query_item.uid] = 0
                self.synthentic_rewarded[query_item.uid] += 1
                self.total_uids_remaining -= 1

            yield category, uids_to_query, should_rewards, time_to_sleep

    def random_should_reward(self, uid):
        if uid not in self.synthentic_rewarded or self.synthentic_rewarded[uid] < 2:
            return True
        return random.random() < 0.2 ## 20% chance of rewarding


    def get_query_for_proxy(self, category):
        synthentic_q = self.synthentic_queue[category]
        proxy_q = self.proxy_queue[category]
        while not synthentic_q.empty():
            query_item = synthentic_q.get()
            should_reward = False
            if (query_item.uid not in self.synthentic_rewarded) or (self.synthentic_rewarded[query_item.uid] <= 20):
                should_reward = True
            yield query_item.uid, should_reward
        while not proxy_q.empty():
            query_item = proxy_q.get()
            yield query_item.uid, False

    def get_rate_limit_by_type(self, rate_limit):
        synthentic_rate_limit = max(1, int(math.floor(rate_limit * 0.8)) - 1)
        synthentic_rate_limit = max(
            rate_limit - synthentic_rate_limit, synthentic_rate_limit
        )
        proxy_rate_limit = rate_limit - synthentic_rate_limit
        return synthentic_rate_limit, proxy_rate_limit