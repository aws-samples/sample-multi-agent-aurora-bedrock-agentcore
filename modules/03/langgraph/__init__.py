# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
Data Visualization Agent Module

This module provides a data visualization agent that can create charts from datasets.
"""

from .dataviz import (
    DataVizAgent,
    DataVizConfig,
    MCPConfigLoader,
    create_dataviz_agent,
    quick_visualize,
    set_chart_config,
    # Export the individual chart creation tools for direct use
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_pie_chart,
    analyze_data_structure,
)

__all__ = [
    'DataVizAgent',
    'DataVizConfig',
    'MCPConfigLoader',
    'create_dataviz_agent',
    'quick_visualize',
    'set_chart_config',
    'create_bar_chart',
    'create_line_chart',
    'create_scatter_plot',
    'create_pie_chart',
    'analyze_data_structure',
]