#!/bin/bash

script_dir=$(cd "$(dirname $0)"; pwd)
yaml_file="$script_dir/.jenkins/test/config/dependent_packages.yaml"
work_dir="install_depend_pkgs"

if [ ! -f "$yaml_file" ]; then
    echo "$yaml_file does not exist."
    exit 1
fi

if [ ! -d "$work_dir" ]; then
    mkdir -p "$work_dir"
    echo "Created $work_dir directory."
else
    echo "$work_dir already exists. Removing existing whl packages."
    rm -f "$work_dir"/*.whl
fi

cd "$work_dir" || exit 1

get_yaml_value() {
    local file="$1"
    local key="$2"

    python3 -c "
import yaml
try:
    with open('$file', 'r') as f:
        data = yaml.safe_load(f)
        print(data.get('$key', ''))
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"
}


python_v="cp$(python3 --version 2>&1 | grep -oP 'Python \K\d+\.\d+' | tr -d .)"
echo "========= Installing vLLM and ray ========="
vllm_path=$(get_yaml_value "$yaml_file" "vllm")
vllm_name="vllm-0.8.4.dev0+g296c657.d20250514.empty-py3-none-any.whl"
vllm_pkg="${vllm_path}any/${vllm_name}"
wget "$vllm_pkg" --no-check-certificate || { echo "Failed to download vllm"; exit 1; }
pip uninstall vllm -y && pip install "$vllm_name" || { echo "Failed to install vllm"; exit 1; }
pip install ray
pip uninstall torch torch-npu torchvision torchaudio -y


echo "========= Installing MSAdapter ========="
msadapter_path=$(get_yaml_value "$yaml_file" "msadapter")
msadapter_name="msadapter-0.0.1-py3-none-any.whl"
msadapter_pkg="${msadapter_path}any/${msadapter_name}"
wget "$msadapter_pkg" --no-check-certificate || { echo "Failed to download vllm"; exit 1; }
pip uninstall msadapter -y && pip install "$msadapter_name" || { echo "Failed to install vllm"; exit 1; }

echo "========= Installing vLLM-MindSpore ========="
vllm_mindspore_path=$(get_yaml_value "$yaml_file" "vllm-mindspore")
vllm_mindspore_name="vllm_mindspore-0.3.0-${python_v}-${python_v}-linux_$(arch).whl"
vllm_mindspore_pkg="${vllm_mindspore_path}ascend/$(arch)/${vllm_mindspore_name}"
wget "$vllm_mindspore_pkg" --no-check-certificate || { echo "Failed to download vllm_mindspore"; exit 1; }
pip uninstall vllm-mindspore -y && pip install "$vllm_mindspore_name" || { echo "Failed to install vllm_ mindspore"; exit 1; }

echo "========= Installing mindspore_gs ========="
mindspore_gs_path=$(get_yaml_value "$yaml_file" "mindspore_gs")
mindspore_gs_name="mindspore_gs-1.2.0.dev20250604-py3-none-any.whl"
mindspore_gs_pkg="${mindspore_gs_path}any/${mindspore_gs_name}"
wget "$mindspore_gs_pkg" --no-check-certificate || { echo "Failed to download mindspore_gs"; exit 1; }
pip uninstall mindspore_gs -y && pip install "$mindspore_gs_name" || { echo "Failed to install mindspore_gs"; exit 1; }

echo "========= Installing mindspore ========="
mindspore_path=$(get_yaml_value "$yaml_file" "mindspore")
mindspore_name="mindspore-2.7.0-${python_v}-${python_v}-linux_$(arch).whl"
mindspore_pkg="${mindspore_path}unified/$(arch)/${mindspore_name}"
wget "$mindspore_pkg" --no-check-certificate || { echo "Failed to download mindspore"; exit 1; }
pip uninstall mindspore -y && pip install "$mindspore_name" || { echo "Failed to install mindspore"; exit 1; }

echo "========= Installing mindformers ========="
mf_dir=mindformers-dev
if [ ! -d "$mf_dir" ]; then
    git clone https://gitee.com/mindspore/mindformers.git -b dev "$mf_dir"
    if [ ! -d "$mf_dir" ]; then
        echo "Failed to git clone mindformers!"
        exit 1
    fi
    cd $mf_dir
    git reset --hard 6a52b43
    pip uninstall mindformers -y
    bash build.sh
else
    echo "The $mf_dir folder already exists and will not be re-downloaded."
fi

echo "All dependencies installed successfully!"
