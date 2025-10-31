#!/usr/bin/env python3
"""
Plot comparison of ground truth vs VLM output counts across video splits.
"""

import os
import json
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


def extract_part_number(filename):
    """Extract the part number from filename like 'episode_0_gt_part_5.json'"""
    match = re.search(r'part_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def load_count_from_json(filepath):
    """Load the 'count' field from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get('count', 0)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def main():
    # Define directories
    gt_dir = "benchmark/spatial_association/ground_truth_split"
    output_dir = "output/split"
    
    # Get all ground truth files
    gt_pattern = os.path.join(gt_dir, "episode_0_gt_part_*.json")
    gt_files = glob.glob(gt_pattern)
    
    # Sort files numerically by part number, not alphabetically
    gt_files.sort(key=lambda f: extract_part_number(f) or 0)
    
    # Extract data
    split_numbers = []
    gt_counts = []
    vlm_counts = []
    
    for gt_file in gt_files:
        part_num = extract_part_number(gt_file)
        if part_num is None:
            continue
        
        # Load ground truth count
        gt_count = load_count_from_json(gt_file)
        if gt_count is None:
            continue
        
        # Find corresponding VLM output file
        vlm_file = os.path.join(output_dir, f"episode_0_720p_10fps_part_{part_num}_combined.json")
        if not os.path.exists(vlm_file):
            print(f"Warning: VLM file not found for part {part_num}")
            continue
        
        vlm_count = load_count_from_json(vlm_file)
        if vlm_count is None:
            continue
        
        # Store data
        split_numbers.append(part_num)
        gt_counts.append(gt_count)
        vlm_counts.append(vlm_count)
    
    if not split_numbers:
        print("No data found to plot!")
        return
    
    # Calculate error (absolute difference)
    errors = [abs(gt - vlm) for gt, vlm in zip(gt_counts, vlm_counts)]
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    n_points = len(split_numbers)
    
    # Plot ground truth as line segments between successive points
    for i in range(n_points - 1):
        ax1.plot(split_numbers[i:i+2], gt_counts[i:i+2], 'o-', 
                 color='darkgreen', linewidth=2.5, markersize=7, alpha=0.8, label='Ground Truth' if i == 0 else '')

    print("\n" + "="*50)
    print("RAW DATA")
    print(list(zip(split_numbers, gt_counts)))
    print(list(zip(split_numbers, vlm_counts)))

    # Plot VLM output as a single solid color
    ax1.plot(split_numbers, vlm_counts, 's-', label='VLM Output', 
             color='blue', linewidth=2.5, markersize=7, alpha=0.8)
    
    # Add polynomial trendline for VLM outputs (degree 2)
    z = np.polyfit(split_numbers, vlm_counts, 2)
    p = np.poly1d(z)
    ax1.plot(split_numbers, p(split_numbers), "--", 
             color='purple', linewidth=2.5, alpha=0.7, label='VLM Polynomial Fit')
    
    # Plot error on the same axis
    ax1.plot(split_numbers, errors, '^-', label='Absolute Error', 
             color='red', linewidth=2, markersize=6, alpha=0.7)
    
    # Add polynomial trendline for error (degree 2)
    z_err = np.polyfit(split_numbers, errors, 2)
    p_err = np.poly1d(z_err)
    ax1.plot(split_numbers, p_err(split_numbers), "--", 
             color='orange', linewidth=2.5, alpha=0.7, label='Error Polynomial Fit')
    
    ax1.set_xlabel('Split Number', fontsize=13)
    ax1.set_ylabel('Count / Error', fontsize=13, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Add more tick labels on both axes
    ax1.xaxis.set_major_locator(MultipleLocator(1))  # Major ticks every 1 unit on x-axis
    ax1.yaxis.set_major_locator(MultipleLocator(1))  # Major ticks every 1 unit on y-axis
    
    # Enable major and minor grid lines
    ax1.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    
    # Title and legends
    plt.title('Cubicle Count Comparison: Ground Truth vs VLM Output Across Video Splits', 
              fontsize=14, pad=20)
    
    # Legend
    ax1.legend(loc='upper left', fontsize=11)

    # Add statistics text box
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    perfect_count = sum(1 for e in errors if e == 0)
    total_count = len(errors)
    accuracy = perfect_count / total_count * 100
    
    stats_text = f'Mean Error: {mean_error:.2f}\nMax Error: {max_error}\nPerfect Matches: {perfect_count} of {total_count} ({accuracy:.1f}%)'
    ax1.text(0.98, 0.03, stats_text, transform=ax1.transAxes,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_plot = os.path.join(output_dir, "count_comparison_plot.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_plot}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total splits analyzed: {len(split_numbers)}")
    print(f"Ground Truth counts - Min: {min(gt_counts)}, Max: {max(gt_counts)}, Mean: {np.mean(gt_counts):.2f}")
    print(f"VLM Output counts - Min: {min(vlm_counts)}, Max: {max(vlm_counts)}, Mean: {np.mean(vlm_counts):.2f}")
    print(f"Perfect matches (error = 0): {sum(1 for e in errors if e == 0)} out of {len(errors)} ({accuracy:.1f}%)")
    print("="*50)


if __name__ == "__main__":
    main()
