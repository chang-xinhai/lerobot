#!/bin/bash

# ==========================================
# LeRobot Checkpoint Manager (Aliyunpan Version)
# ==========================================

# --- Path Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_PATH="$PROJECT_ROOT/outputs/train"

# Aliyunpan Configuration
# Ensure 'aliyunpan' is in your PATH, or set the full path here (e.g., /home/xinhai/bin/aliyunpan)
ALI_CMD="aliyunpan"
REMOTE_BASE="/Research/automoma/ckpt/lerobot"

# --- Colors ---
CLR_ACT='\033[0;32m'     # Green
CLR_DP='\033[0;31m'      # Red
CLR_DP3='\033[0;36m'     # Cyan
CLR_SMOL='\033[0;35m'    # Purple
CLR_WARN='\033[1;33m'    # Yellow
CLR_RESET='\033[0m'      # Reset

# --- Argument Parsing ---
MODE=$1          # upload | download
POLICY_ARG=$2    # act | dp | dp3 | smolvla | all
KEYWORD_ARG=$3   # keyword (e.g., Arena) or "all"
DRY_RUN=$4       # --dry-run (optional)

if [[ -z "$MODE" || -z "$POLICY_ARG" || -z "$KEYWORD_ARG" ]]; then
    echo "Usage: bash scripts/manage_checkpoints_aliyun.sh [upload|download] [policy|all] [keyword|all] [--dry-run]"
    exit 1
fi

IS_DRY=false
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    IS_DRY=true
    echo -e "\n${CLR_WARN}⚠️  DRY-RUN MODE: No files will be moved.${CLR_RESET}"
fi

if [ "$POLICY_ARG" == "all" ]; then
    POLICIES=("act" "dp" "dp3" "smolvla")
else
    POLICIES=("$POLICY_ARG")
fi

# --- Helper: Check Aliyunpan Connectivity ---
check_aliyun() {
    if ! command -v $ALI_CMD &> /dev/null; then
        echo -e "${CLR_DP}Error: '$ALI_CMD' command not found.${CLR_RESET}"
        exit 1
    fi
}

# --- Helper: Select Files to Sync (Local) ---
get_files_to_sync() {
    local folder_path=$1
    local policy_type=$2
    local -n arr_ref=$3 

    # 1. Check if 'checkpoints' folder exists
    local CKPT_DIR="$folder_path/checkpoints"
    if [ ! -d "$CKPT_DIR" ]; then CKPT_DIR="$folder_path"; fi

    # 2. Find the directory with the largest number
    local LATEST_CKPT=$(ls -F "$CKPT_DIR" 2>/dev/null | grep '/' | grep -E '^[0-9]+' | sed 's|/||g' | sort -V | tail -n 1)

    if [ -n "$LATEST_CKPT" ]; then
        arr_ref+=("$CKPT_DIR/$LATEST_CKPT")
        [ "$IS_DRY" = false ] && echo -e "   ${CLR_WARN}💡 Found latest checkpoint: $LATEST_CKPT${CLR_RESET}"
    else
        # Fallback to 'last' or 'pretrained'
        if [ -d "$CKPT_DIR/last" ]; then
            arr_ref+=("$CKPT_DIR/last")
            [ "$IS_DRY" = false ] && echo -e "   ${CLR_WARN}💡 Found 'last' checkpoint.${CLR_RESET}"
        elif [ -d "$CKPT_DIR/pretrained" ]; then
             arr_ref+=("$CKPT_DIR/pretrained")
        fi
    fi

    # 3. Include config files
    if [ -f "$folder_path/config.yaml" ]; then arr_ref+=("$folder_path/config.yaml"); fi
    if [ -f "$folder_path/hydra_config.yaml" ]; then arr_ref+=("$folder_path/hydra_config.yaml"); fi
}

# --- Action: Process Single Folder ---
process_folder() {
    local pol=$1
    local task_name=$2
    
    # Determine Color
    local p_color=$CLR_RESET
    case $pol in
        act) p_color=$CLR_ACT ;;
        dp)  p_color=$CLR_DP ;;
        dp3) p_color=$CLR_DP3 ;;
        smolvla) p_color=$CLR_SMOL ;;
    esac

    echo "-----------------------------------------------"
    echo -e "📂 Task: [${p_color}${pol^^}${CLR_RESET}] $task_name"

    if [ "$MODE" == "upload" ]; then
        local full_local_path="$BASE_PATH/$task_name"
        local FILES_TO_SYNC=()
        get_files_to_sync "$full_local_path" "$pol" FILES_TO_SYNC

        if [ ${#FILES_TO_SYNC[@]} -eq 0 ]; then
            echo -e "   \033[0;35m❌ Skip: No valid checkpoints found.\033[0m"
            return
        fi

        if [ "$IS_DRY" = true ]; then
            echo "   📝 [Dry-Run] Upload List:"
            for f in "${FILES_TO_SYNC[@]}"; do echo "      - $(basename "$f")"; done
            echo -e "   📡 To Remote: $REMOTE_BASE/$task_name"
        else
            # Stage files to preserve structure: /tmp/unique_id/task_name/checkpoints/xxx
            local TMP_ROOT="/tmp/ali_sync_$(date +%s%N)"
            local TMP_TASK_DIR="$TMP_ROOT/$task_name"
            
            mkdir -p "$TMP_TASK_DIR"

            for f in "${FILES_TO_SYNC[@]}"; do 
                base_name=$(basename "$f")
                if [[ "$base_name" == *".yaml" ]]; then
                    cp "$f" "$TMP_TASK_DIR/"
                else 
                    mkdir -p "$TMP_TASK_DIR/checkpoints"
                    cp -r "$f" "$TMP_TASK_DIR/checkpoints/"
                fi
            done
            
            echo -e "   🚀 Uploading via Aliyunpan..."
            # Create remote base if needed (aliyunpan usually handles this, but mkdir is safer)
            # $ALI_CMD mkdir "$REMOTE_BASE" > /dev/null 2>&1 

            # Upload the TASK FOLDER into the REMOTE BASE
            # Result: /Remote/Base/task_name/...
            $ALI_CMD upload "$TMP_TASK_DIR" "$REMOTE_BASE"
            
            rm -rf "$TMP_ROOT"
            echo -e "   ${CLR_ACT}✅ Done.${CLR_RESET}"
        fi

    elif [ "$MODE" == "download" ]; then
        if [ "$IS_DRY" = true ]; then
             echo -e "   📝 [Dry-Run] Download: $REMOTE_BASE/$task_name"
             echo -e "   📍 To Local: $BASE_PATH/$task_name"
        else
             echo -e "   📥 Downloading..."
             mkdir -p "$BASE_PATH" # Ensure parent exists
             # Aliyunpan download syntax: download <remote_file_or_dir> <local_dir>
             $ALI_CMD download "$REMOTE_BASE/$task_name" "$BASE_PATH"
             echo -e "   ${CLR_ACT}✅ Done.${CLR_RESET}"
        fi
    fi
}

# --- Main Logic ---

check_aliyun

# 1. Gather Targets (Local or Remote)
if [ "$MODE" == "upload" ]; then
    if [ ! -d "$BASE_PATH" ]; then
        echo -e "\n${CLR_DP}Error: Base path does not exist: $BASE_PATH${CLR_RESET}"
        exit 1
    fi
    # Get local directory names
    ALL_DIRS=$(ls -d "$BASE_PATH"/*/ 2>/dev/null | xargs -n 1 basename)

elif [ "$MODE" == "download" ]; then
    echo -e "\n🔍 Scanning Remote Directory ($REMOTE_BASE)..."
    # Parse Aliyunpan 'ls' output.
    # Typical output:
    # #  FileSize  Date       Time     Name
    # 1  -         2026-01-12 11:54:48 my_folder/
    # We strip the trailing slash and ignore headers
    
    RAW_LS=$($ALI_CMD ls "$REMOTE_BASE")
    
    # Extract the last column ($NF), remove trailing slash, ignore lines starting with non-text
    ALL_DIRS=$(echo "$RAW_LS" | awk '{print $NF}' | sed 's|/$||g' | grep -v "文件(目录)" | grep -v "\-\-\-\-" | grep -v "总:" | grep -v "当前目录")
fi

# 2. Loop Policies and Filter
for pol in "${POLICIES[@]}"; do
    # Filter folders that start with {policy}_
    TARGETS=$(echo "$ALL_DIRS" | grep "^${pol}_")
    
    # Filter by Keyword
    if [ "$KEYWORD_ARG" != "all" ]; then
        TARGETS=$(echo "$TARGETS" | grep "$KEYWORD_ARG")
    fi

    if [ -z "$TARGETS" ]; then
        continue
    fi

    for t in $TARGETS; do
        process_folder "$pol" "$t"
    done
done

echo -e "\n✨ Operation Complete.\n"