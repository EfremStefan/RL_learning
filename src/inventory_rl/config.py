from dataclasses import dataclass, replace


@dataclass(frozen=True)
class InventoryConfig:
    """Parameters of the lost-sales inventory problem.

    An order placed today is delivered at the start of day t + L, so it can
    first be sold L days after it was placed.
    """

    L: int = 1          # lead time in days
    lam: float = 4.0    # mean daily Poisson demand
    X_max: int = 20     # shelf capacity; deliveries beyond it are discarded
    A_max: int = 10     # largest order per day
    h: float = 1.0      # holding cost per unit left at the end of the day
    p: float = 10.0     # penalty per unit of lost sales
    K: float = 5.0      # fixed cost per order placed
    c: float = 1.0      # purchase cost per unit ordered
    gamma: float = 0.99

    def __post_init__(self):
        if self.L < 1:
            raise ValueError("lead time L must be at least 1")

    def with_(self, **changes) -> "InventoryConfig":
        return replace(self, **changes)

    @property
    def pipeline_len(self) -> int:
        return self.L - 1

    @property
    def n_actions(self) -> int:
        return self.A_max + 1

    @property
    def state_dims(self) -> tuple[int, ...]:
        return (self.X_max + 1,) + (self.A_max + 1,) * self.pipeline_len

    @property
    def n_states(self) -> int:
        n = 1
        for d in self.state_dims:
            n *= d
        return n

    @property
    def max_position(self) -> int:
        return self.X_max + self.pipeline_len * self.A_max


DEFAULT = InventoryConfig()

LEAD_TIMES = range(1, 7)


def lead_time_sweep(base: InventoryConfig = DEFAULT, lead_times=LEAD_TIMES):
    return {L: base.with_(L=L) for L in lead_times}
