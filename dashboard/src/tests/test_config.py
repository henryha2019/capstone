import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.config import (
    RATINGS, RATING_DESCRIPTIONS, LOCATION_COLOUR_MAP, 
    LOCATION_COLOUR_EMOJI, RATING_COLOUR_MAP, RATING_COLOUR_EMOJI, FEATURES
)


class TestConfigStructure:
    
    def test_ratings_has_required_keys(self):
        assert "status_cols" in RATINGS
        assert "metric_cols" in RATINGS
        assert isinstance(RATINGS["status_cols"], list)
        assert isinstance(RATINGS["metric_cols"], list)
    
    def test_ratings_columns_are_strings(self):
        all_ratings = RATINGS["status_cols"] + RATINGS["metric_cols"]
        assert all(isinstance(rating, str) for rating in all_ratings)
        assert len(all_ratings) > 0
    
    def test_rating_descriptions_covers_all_ratings(self):
        all_ratings = RATINGS["status_cols"] + RATINGS["metric_cols"]
        missing_descriptions = set(all_ratings) - set(RATING_DESCRIPTIONS.keys())
        
        assert len(missing_descriptions) == 0, f"Missing descriptions for: {missing_descriptions}"
    
    def test_rating_colour_maps_cover_all_ratings(self):
        all_ratings = RATINGS["status_cols"] + RATINGS["metric_cols"]
        missing_colors = set(all_ratings) - set(RATING_COLOUR_MAP.keys())
        missing_emojis = set(all_ratings) - set(RATING_COLOUR_EMOJI.keys())
        
        assert len(missing_colors) == 0, f"Missing colors for: {missing_colors}"
        assert len(missing_emojis) == 0, f"Missing emojis for: {missing_emojis}"


class TestColorMaps:
    
    def test_location_color_map_values_are_hex_colors(self):
        for location, color in LOCATION_COLOUR_MAP.items():
            assert isinstance(color, str)
            assert color.startswith('#')
            assert len(color) == 7
    
    def test_rating_color_map_values_are_hex_colors(self):
        for rating, color in RATING_COLOUR_MAP.items():
            assert isinstance(color, str)
            assert color.startswith('#')
            assert len(color) == 7
    
    def test_emoji_maps_have_string_values(self):
        for emoji in LOCATION_COLOUR_EMOJI.values():
            assert isinstance(emoji, str)
            assert len(emoji) > 0
        
        for emoji in RATING_COLOUR_EMOJI.values():
            assert isinstance(emoji, str)
            assert len(emoji) > 0
    
    def test_location_maps_have_matching_keys(self):
        color_keys = set(LOCATION_COLOUR_MAP.keys())
        emoji_keys = set(LOCATION_COLOUR_EMOJI.keys())
        assert color_keys == emoji_keys


class TestFeatures:
    
    def test_features_has_vibration_velocity_key(self):
        assert "vibration_velocity_z" in FEATURES
        assert isinstance(FEATURES["vibration_velocity_z"], str)
        assert len(FEATURES["vibration_velocity_z"]) > 0
    
    def test_features_values_are_column_names(self):
        for feature_key, column_name in FEATURES.items():
            assert isinstance(feature_key, str)
            assert isinstance(column_name, str)
            assert len(column_name) > 0


class TestConfigConsistency:
    
    def test_no_duplicate_ratings(self):
        status_cols = RATINGS["status_cols"]
        metric_cols = RATINGS["metric_cols"]
        
        assert len(set(status_cols)) == len(status_cols)
        assert len(set(metric_cols)) == len(metric_cols)
        assert len(set(status_cols) & set(metric_cols)) == 0
    
    def test_rating_descriptions_have_content(self):
        for rating, description in RATING_DESCRIPTIONS.items():
            assert isinstance(description, str)
            assert len(description.strip()) > 0
    
    def test_all_configs_exist(self):
        configs = [RATINGS, RATING_DESCRIPTIONS, LOCATION_COLOUR_MAP, 
                  LOCATION_COLOUR_EMOJI, RATING_COLOUR_MAP, RATING_COLOUR_EMOJI, FEATURES]
        
        for config in configs:
            assert config is not None
            assert len(config) > 0