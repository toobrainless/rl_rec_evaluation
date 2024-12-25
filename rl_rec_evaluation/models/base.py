from abc import ABC

class BaseModel(ABC):
    def state_batch(self, batch):
        pass

    def score_batch(self, batch):
        pass

    def score_with_state(self, batch):
        pass
