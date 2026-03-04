import pytest
from roles import select_roles, AGENT_ROLES


class TestRoleSelection:
    def test_select_fewer_than_defined(self):
        configs = select_roles(3)
        assert len(configs) == 3
        assert configs[0]["label"] == "methodical"
        assert configs[1]["label"] == "creative"
        assert configs[2]["label"] == "skeptical"

    def test_select_exact_defined_count(self):
        configs = select_roles(len(AGENT_ROLES))
        assert len(configs) == len(AGENT_ROLES)
        labels = [c["label"] for c in configs]
        assert labels == [r["name"] for r in AGENT_ROLES]

    def test_select_more_than_defined_has_stochastic_overflow(self):
        configs = select_roles(7)
        assert len(configs) == 7
        # First 5 are defined roles
        for i in range(len(AGENT_ROLES)):
            assert configs[i]["label"] == AGENT_ROLES[i]["name"]
        # Extras are stochastic
        assert configs[5]["label"].startswith("stochastic_")
        assert configs[6]["label"].startswith("stochastic_")

    def test_all_configs_have_required_keys(self):
        configs = select_roles(7)
        required_keys = {"label", "temperature", "top_p", "planner_addendum", "executor_addendum"}
        for cfg in configs:
            assert required_keys.issubset(cfg.keys()), f"Missing keys in {cfg['label']}"

    def test_temperature_in_valid_range(self):
        configs = select_roles(10)
        for cfg in configs:
            assert 0.01 <= cfg["temperature"] <= 2.0, f"Bad temp for {cfg['label']}"
            assert 0.1 <= cfg["top_p"] <= 1.0, f"Bad top_p for {cfg['label']}"

    def test_select_zero(self):
        configs = select_roles(0)
        assert configs == []

    def test_defined_roles_have_addendums(self):
        configs = select_roles(5)
        for cfg in configs:
            assert cfg["planner_addendum"], f"{cfg['label']} has empty planner_addendum"
            assert cfg["executor_addendum"], f"{cfg['label']} has empty executor_addendum"

    def test_stochastic_overflow_has_empty_addendums(self):
        configs = select_roles(6)
        overflow = configs[5]
        assert overflow["planner_addendum"] == ""
        assert overflow["executor_addendum"] == ""
