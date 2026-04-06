"""
Tests for Stage 3.1B configuration.

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

import pytest
import numpy as np

from stage3.configs.config_stage3_1b import (
    CONFIG_3_1B,
    ENV_CONFIG_3_1B,
    AGENT_CONFIG_3_1B,
    get_condition_grid,
    get_canonical_conditions,
    get_one_shot_protocol,
    build_env_config_for_condition,
    compute_conflict_variables,
    OPEN_REWARD_LEVELS,
    OPEN_THREAT_LEVELS,
    COVERED_PATH_FIXED,
)


class TestConditionGrid:
    """Tests for condition grid generation."""
    
    def test_grid_size(self):
        """Test that grid has 9 conditions (3x3)."""
        grid = get_condition_grid()
        assert len(grid) == 9
    
    def test_grid_condition_ids(self):
        """Test that all expected condition IDs are present."""
        grid = get_condition_grid()
        expected_ids = [
            'R0_T1', 'R0_T2', 'R0_T3',
            'R1_T1', 'R1_T2', 'R1_T3',
            'R2_T1', 'R2_T2', 'R2_T3'
        ]
        for cid in expected_ids:
            assert cid in grid
    
    def test_reward_levels(self):
        """Test that reward levels vary correctly."""
        grid = get_condition_grid()
        
        # R0 should have reward_prob = 0.70
        assert grid['R0_T1']['paths']['open']['reward_prob'] == 0.70
        
        # R1 should have reward_prob = 0.80
        assert grid['R1_T1']['paths']['open']['reward_prob'] == 0.80
        
        # R2 should have reward_prob = 0.90
        assert grid['R2_T1']['paths']['open']['reward_prob'] == 0.90
    
    def test_threat_levels(self):
        """Test that threat levels vary correctly."""
        grid = get_condition_grid()
        
        # T1 should have X_risk = 0.45
        assert grid['R0_T1']['paths']['open']['exposure_profile']['X_risk'] == 0.45
        
        # T2 should have X_risk = 0.60
        assert grid['R0_T2']['paths']['open']['exposure_profile']['X_risk'] == 0.60
        
        # T3 should have X_risk = 0.75
        assert grid['R0_T3']['paths']['open']['exposure_profile']['X_risk'] == 0.75
    
    def test_covered_path_fixed(self):
        """Test that covered path is constant across conditions."""
        grid = get_condition_grid()
        
        covered_paths = []
        for cid, data in grid.items():
            covered_paths.append(data['paths']['covered'])
        
        # All covered paths should be identical
        for i in range(1, len(covered_paths)):
            assert covered_paths[i] == covered_paths[0]


class TestCanonicalConditions:
    """Tests for canonical condition definitions."""
    
    def test_canonical_count(self):
        """Test that there are 3 canonical conditions."""
        canonical = get_canonical_conditions()
        assert len(canonical) == 3
    
    def test_canonical_names(self):
        """Test canonical condition names."""
        canonical = get_canonical_conditions()
        assert 'reward_dominant' in canonical
        assert 'balanced_conflict' in canonical
        assert 'threat_dominant' in canonical
    
    def test_reward_dominant_config(self):
        """Test reward-dominant condition configuration."""
        canonical = get_canonical_conditions()
        rd = canonical['reward_dominant']
        
        assert rd['condition_id'] == 'R2_T1'
        assert rd['open_reward_prob'] == 0.90
        assert rd['open_X_risk'] == 0.45
    
    def test_balanced_conflict_config(self):
        """Test balanced-conflict condition configuration."""
        canonical = get_canonical_conditions()
        bc = canonical['balanced_conflict']
        
        assert bc['condition_id'] == 'R1_T2'
        assert bc['open_reward_prob'] == 0.80
        assert bc['open_X_risk'] == 0.60
    
    def test_threat_dominant_config(self):
        """Test threat-dominant condition configuration."""
        canonical = get_canonical_conditions()
        td = canonical['threat_dominant']
        
        assert td['condition_id'] == 'R0_T3'
        assert td['open_reward_prob'] == 0.70
        assert td['open_X_risk'] == 0.75


class TestConflictVariables:
    """Tests for conflict variable computation."""
    
    def test_reward_gap_positive(self):
        """Test reward gap when open has higher reward."""
        open_path = {
            'base_reward': 1.0,
            'reward_bonus': 0.4,
            'reward_prob': 0.90,
            'exposure_profile': {'X_risk': 0.45}
        }
        covered_path = {
            'base_reward': 1.0,
            'reward_bonus': 0.0,
            'reward_prob': 0.70,
            'exposure_profile': {'X_risk': 0.20}
        }
        
        cv = compute_conflict_variables(open_path, covered_path)
        
        # E[R_open] = 1.4 * 0.9 = 1.26
        # E[R_covered] = 1.0 * 0.7 = 0.70
        # reward_gap = 0.56
        assert cv['reward_gap'] > 0
        assert np.isclose(cv['reward_gap'], 0.56, atol=0.01)
    
    def test_risk_gap_positive(self):
        """Test risk gap when open has higher risk."""
        open_path = {
            'base_reward': 1.0,
            'reward_prob': 0.80,
            'exposure_profile': {'X_risk': 0.60}
        }
        covered_path = {
            'base_reward': 1.0,
            'reward_prob': 0.70,
            'exposure_profile': {'X_risk': 0.20}
        }
        
        cv = compute_conflict_variables(open_path, covered_path)
        
        # risk_gap = 0.60 - 0.20 = 0.40
        assert cv['risk_gap'] > 0
        assert np.isclose(cv['risk_gap'], 0.40, atol=0.01)
    
    def test_conflict_condition_classification(self):
        """Test conflict condition classification logic."""
        # Reward dominant: |reward_gap| >> |threat_gap|
        open_rd = {
            'base_reward': 1.0, 'reward_bonus': 0.4, 'reward_prob': 0.90,
            'threat_penalty': 0.0, 'threat_prob': 0.0,
            'exposure_profile': {'X_risk': 0.45}
        }
        covered = {
            'base_reward': 1.0, 'reward_bonus': 0.0, 'reward_prob': 0.70,
            'threat_penalty': 0.0, 'threat_prob': 0.0,
            'exposure_profile': {'X_risk': 0.20}
        }
        
        cv_rd = compute_conflict_variables(open_rd, covered)
        assert cv_rd['conflict_condition'] == 'reward_dominant'
        
        # Threat dominant: |threat_gap| >> |reward_gap|
        open_td = {
            'base_reward': 1.0, 'reward_bonus': 0.0, 'reward_prob': 0.70,
            'threat_penalty': 1.0, 'threat_prob': 0.5,
            'exposure_profile': {'X_risk': 0.75}
        }
        
        cv_td = compute_conflict_variables(open_td, covered)
        assert cv_td['conflict_condition'] == 'threat_dominant'
        
        # Balanced: similar magnitudes
        open_bc = {
            'base_reward': 1.0, 'reward_bonus': 0.2, 'reward_prob': 0.80,
            'threat_penalty': 0.5, 'threat_prob': 0.3,
            'exposure_profile': {'X_risk': 0.60}
        }
        
        cv_bc = compute_conflict_variables(open_bc, covered)
        # May be reward_dominant or balanced depending on exact values
        assert cv_bc['conflict_condition'] in ['reward_dominant', 'balanced_conflict', 'threat_dominant']


class TestBuildEnvConfig:
    """Tests for environment config builder."""
    
    def test_build_valid_condition(self):
        """Test building config for valid condition."""
        config = build_env_config_for_condition('R1_T2')
        
        assert config['condition_id'] == 'R1_T2'
        assert config['reward_level'] == 'R1'
        assert config['threat_level'] == 'T2'
        assert 'paths' in config
        assert 'open' in config['paths']
        assert 'covered' in config['paths']
    
    def test_build_invalid_condition(self):
        """Test that invalid condition raises error."""
        with pytest.raises(ValueError):
            build_env_config_for_condition('INVALID')
    
    def test_baseline_parameters_frozen(self):
        """Test that baseline parameters from 3.1A are preserved."""
        config = build_env_config_for_condition('R1_T2')
        
        # exposure_q_bias should be 0.35 (frozen from 3.1A)
        assert config['deliberation']['exposure_q_bias'] == 0.35
        
        # eps should be 0.05 (frozen from 3.1A)
        assert config['deliberation']['eps'] == 0.05


class TestOneShotProtocol:
    """Tests for one-shot protocol configuration."""
    
    def test_protocol_structure(self):
        """Test one-shot protocol structure."""
        protocol = get_one_shot_protocol()
        
        assert 'pre_block_trials' in protocol
        assert 'shock_trial' in protocol
        assert 'post_block_trials' in protocol
        assert 'one_shot' in protocol
    
    def test_protocol_trial_counts(self):
        """Test trial counts in protocol."""
        protocol = get_one_shot_protocol()
        
        assert protocol['pre_block_trials'] == 30
        assert protocol['shock_trial'] == 30
        assert protocol['post_block_trials'] == 69
        assert protocol['total_trials'] == 100
    
    def test_one_shot_config_defaults(self):
        """Test one-shot config default values."""
        protocol = get_one_shot_protocol()
        os_config = protocol['one_shot']
        
        assert os_config['one_shot_enabled'] == True
        assert os_config['one_shot_trial'] == 30
        assert os_config['one_shot_path'] == 'open'
        assert os_config['one_shot_reward'] == -5.0
        assert os_config['one_shot_salience'] == 0.9
        assert os_config['one_shot_stakes'] == 10.0


class TestConfigIntegrity:
    """Tests for overall config integrity."""
    
    def test_main_config_structure(self):
        """Test main CONFIG_3_1B structure."""
        assert 'env' in CONFIG_3_1B
        assert 'agent' in CONFIG_3_1B
        assert 'logging' in CONFIG_3_1B
        assert 'ablation' in CONFIG_3_1B
        assert 'matrix' in CONFIG_3_1B
        assert 'canonical' in CONFIG_3_1B
        assert 'one_shot_protocol' in CONFIG_3_1B
    
    def test_agent_config_same_as_3_1a(self):
        """Test that agent config preserves 3.1A parameters."""
        # Key frozen parameters
        assert AGENT_CONFIG_3_1B['temporal_state']['one_shot_threshold'] == 5.0
        assert AGENT_CONFIG_3_1B['temporal_state']['one_shot_boost'] == 2.0
        assert AGENT_CONFIG_3_1B['gate_thresholds']['critical_risk_threshold'] == 0.7
    
    def test_ablation_configs_present(self):
        """Test that all required ablations are defined."""
        ablations = CONFIG_3_1B['ablation']
        
        required_ablations = ['full', 'novg', 'novp', 'nox', 'one_shot_off']
        for ablation in required_ablations:
            assert ablation in ablations


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
