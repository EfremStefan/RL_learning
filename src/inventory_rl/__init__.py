"""P30: inventory control with lost sales and order lead time."""

from gymnasium.envs.registration import register

from .config import InventoryConfig

register(id="Inventory-v0", entry_point="inventory_rl.env:InventoryEnv")

__all__ = ["InventoryConfig"]
