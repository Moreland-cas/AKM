#!/bin/bash

# 限制 P 核（0-7）最大频率为 5 GHz
echo "Setting max frequency for P cores (0-7) to 5 GHz..."
for core in {0..7}; do
    sudo cpupower -c $core frequency-set -u 5000MHz
    # sudo cpupower -c $core frequency-set -u 4500MHz
done

# 限制 E 核（8-15）最大频率为 4 GHz
echo "Setting max frequency for E cores (8-15) to 4 GHz..."
for core in {8..15}; do
    sudo cpupower -c $core frequency-set -u 4000MHz
    # sudo cpupower -c $core frequency-set -u 3500MHz
done

# 验证设置
echo "Verifying the frequency settings..."
cpufreq-info
# watch -n 0.5 -d cpufreq-info

echo "Frequency settings applied successfully."
