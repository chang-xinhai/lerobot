#!/bin/bash

# ==========================================
# LeRobot Checkpoint Manager (Updated)
# ==========================================

# --- Path Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up one level from scripts/ to get project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_PATH="$PROJECT_ROOT/outputs/train"
REMOTE_BASE="123pan:/automoma/ckpt/lerobot"

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
    echo "Usage: bash scripts/manage_checkpoints.sh [upload|download] [policy|all] [keyword|all] [--dry-run]"
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

# --- Helper: Select Files to Sync ---
get_files_to_sync() {
    local folder_path=$1
    local policy_type=$2
    local -n arr_ref=$3 # Return array by reference

    # 1. Check if 'checkpoints' folder exists
    local CKPT_DIR="$folder_path/checkpoints"
    
    if [ ! -d "$CKPT_DIR" ]; then
        # Try finding in root if not in subdir (fallback)
        CKPT_DIR="$folder_path"
    fi

    # 2. Find the directory with the largest number
    # We look for directories that are just numbers (regex ^[0-9]+$)
    # We use sort -V (version sort) to handle 001000 vs 005000 correctly
    local LATEST_CKPT=$(ls -F "$CKPT_DIR" 2>/dev/null | grep '/' | grep -E '^[0-9]+' | sed 's|/||g' | sort -V | tail -n 1)

    if [ -n "$LATEST_CKPT" ]; then
        arr_ref+=("$CKPT_DIR/$LATEST_CKPT")
        if [ "$IS_DRY" = false ]; then
             echo -e "   ${CLR_WARN}💡 Found latest checkpoint: $LATEST_CKPT${CLR_RESET}"
        fi
    else
        # If no numbers found, check for 'last' or 'pretrained'
        if [ -d "$CKPT_DIR/last" ]; then
            arr_ref+=("$CKPT_DIR/last")
            [ "$IS_DRY" = false ] && echo -e "   ${CLR_WARN}💡 Found 'last' checkpoint.${CLR_RESET}"
        elif [ -d "$CKPT_DIR/pretrained" ]; then
             arr_ref+=("$CKPT_DIR/pretrained")
        fi
    fi

    # 3. Always include config.yaml if it exists (usually in the task root)
    if [ -f "$folder_path/config.yaml" ]; then
        arr_ref+=("$folder_path/config.yaml")
    elif [ -f "$folder_path/hydra_config.yaml" ]; then
        arr_ref+=("$folder_path/hydra_config.yaml")
    fi
}

# --- Action: Process Single Folder ---
process_folder() {
    local pol=$1
    local folder_name=$2
    local local_root=$3
    local remote_root=$4
    local full_local_path="$local_root/$folder_name"
    
    # Color selection
    local p_color=$CLR_RESET
    case $pol in
        act) p_color=$CLR_ACT ;;
        dp)  p_color=$CLR_DP ;;
        dp3) p_color=$CLR_DP3 ;;
        smolvla) p_color=$CLR_SMOL ;;
    esac

    echo "-----------------------------------------------"
    echo -e "📂 Target: [${p_color}${pol^^}${CLR_RESET}] $folder_name"

    FILES_TO_SYNC=()
    get_files_to_sync "$full_local_path" "$pol" FILES_TO_SYNC

    if [ ${#FILES_TO_SYNC[@]} -eq 0 ]; then
        echo -e "   \033[0;35m❌ Skip: No valid checkpoints found.\033[0m"
        return
    fi

    if [ "$IS_DRY" = true ]; then
        echo "   📝 [Dry-Run] Upload List:"
        for f in "${FILES_TO_SYNC[@]}"; do echo "      - $(basename "$f")"; done
        echo -e "   📡 To Remote: $remote_root/$folder_name/checkpoints"
    else
        # We need to maintain structure on remote: {task_name}/checkpoints/{ckpt_number}
        # Create a temp dir that mimics the structure we want to upload
        TMP_DIR="/tmp/lerobot_sync_$(date +%s%N)"
        
        for f in "${FILES_TO_SYNC[@]}"; do 
            base_name=$(basename "$f")
            
            # If it is a config file, put it in root of task
            if [[ "$base_name" == *".yaml" ]]; then
                mkdir -p "$TMP_DIR"
                cp "$f" "$TMP_DIR/"
            else 
                # If it is a checkpoint folder (e.g. 005000), put it in checkpoints/ subdir
                mkdir -p "$TMP_DIR/checkpoints"
                cp -r "$f" "$TMP_DIR/checkpoints/"
            fi
        done
        
        echo -e "   🚀 Uploading..."
        # Upload contents of TMP_DIR to REMOTE/folder_name
        rclone copy "$TMP_DIR" "$remote_root/$folder_name" -P
        rm -rf "$TMP_DIR"
        echo -e "   ${CLR_ACT}✅ Done.${CLR_RESET}"
    fi
}

# --- Main Logic ---

# 1. Scan Local Directory
ALL_DIRS=$(ls -d "$BASE_PATH"/*/ 2>/dev/null | xargs -n 1 basename)

for pol in "${POLICIES[@]}"; do
    if [ "$MODE" == "upload" ]; then
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
            process_folder "$pol" "$t" "$BASE_PATH" "$REMOTE_BASE"
        done

    elif [ "$MODE" == "download" ]; then
        echo -e "\n🔍 Scanning Remote for [${pol^^}]..."
        # Note: This part assumes Remote has same structure {task_name}/checkpoints/...
        R_LIST=$(rclone lsf "$REMOTE_BASE" --dirs-only 2>/dev/null | sed 's|/||g')
        R_TARGETS=$(echo "$R_LIST" | grep "^${pol}_")

        if [ "$KEYWORD_ARG" != "all" ]; then
            R_TARGETS=$(echo "$R_TARGETS" | grep "$KEYWORD_ARG")
        fi

        for rt in $R_TARGETS; do
            if [ "$IS_DRY" = true ]; then
                 echo -e "   📝 [Dry-Run] Download: $rt -> $BASE_PATH/$rt"
            else
                 echo -e "   📥 Downloading $rt..."
                 mkdir -p "$BASE_PATH/$rt"
                 rclone copy "$REMOTE_BASE/$rt" "$BASE_PATH/$rt" -P
            fi
        done
    fi
done

echo -e "\n✨ Operation Complete.\n"