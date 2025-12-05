#!/usr/bin/env python
"""Test script to verify candidate pool visibility in running demo."""
import time

import requests

# import json


def test_candidate_visibility():
    """Test that candidate pool becomes active during candidate phases."""

    base_url = _print_output_report_heading("Testing candidate pool visibility...", "http://localhost:8050")
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        print(f"✓ Server is running: {response.json()}")
    except Exception as e:
        print(f"✗ Server not running: {e}")
        print("Please start demo mode with: ./demo")
        return

    seen_candidate_phase = _print_output_report_heading(
        "\nMonitoring training state for candidate pool activity...", False
    )
    max_checks = 30

    for i in range(max_checks):  # sourcery skip: no-loop-in-tests
        try:
            response = requests.get(f"{base_url}/api/state", timeout=2)
            data = response.json()

            epoch = data.get("current_epoch", 0)
            phase = data.get("phase", "Unknown")
            pool_status = data.get("candidate_pool_status", "N/A")
            pool_size = data.get("candidate_pool_size", 0)
            top_cand = data.get("top_candidate_id", "")

            print(f"\nCheck {i + 1}: Epoch {epoch}, Phase: {phase}")
            print(f"  Pool Status: {pool_status}, Size: {pool_size}")

            if pool_status == "Active":
                seen_candidate_phase = True
                print("  ✓ CANDIDATE POOL ACTIVE!")
                print(f"  Top Candidate: {top_cand} (score: {data.get('top_candidate_score', 0.0):.4f})")
                candidate_runnerup = data.get("second_candidate_score", 0.0)
                print(f"  Second Candidate: {data.get('second_candidate_id', '')} (score: {candidate_runnerup:.4f})")

                if pool_metrics := data.get("pool_metrics", {}):
                    print("  Pool Metrics:")
                    print(f"    Avg Loss: {pool_metrics.get('avg_loss', 0.0):.4f}")
                    print(f"    Avg Accuracy: {pool_metrics.get('avg_accuracy', 0.0):.4f}")
                    print(f"    Avg F1 Score: {pool_metrics.get('avg_f1_score', 0.0):.4f}")

                # Wait a bit more to see it in action
                time.sleep(3)
                break

            time.sleep(1)

        except Exception as e:
            print(f"  Error: {e}")
            break

    print("\n" + "=" * 60)
    if seen_candidate_phase:
        print("✓ SUCCESS: Candidate pool was activated and data is visible!")
    else:
        print("✗ FAILED: Candidate pool never activated in {max_checks} seconds")
        print("   Note: Candidate phases occur every 5 epochs")
        print("   With 1-second update interval, should see by epoch 10 (~10 seconds)")
    print("=" * 60)


def _print_output_report_heading(arg0, arg1):
    print(arg0)
    print("=" * 60)
    return arg1


if __name__ == "__main__":
    test_candidate_visibility()
