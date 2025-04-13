from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def generate_pairs(line, window_size):
    line = np.array(line)
    line = line[:, 0]

    seqs = []
    for i in range(0, len(line), window_size):
        seq = line[i:i + window_size]
        seqs.append(seq)
    seqs += []
    seq_pairs = []
    for i in range(1, len(seqs)):
        seq_pairs.append([seqs[i - 1], seqs[i]])
    return seqs


def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0):
    # Handle empty lines or lines with just whitespace
    if not line or line.isspace():
        return [], []
        
    # Split the line into tokens
    try:
        line = [ln.split(",") for ln in line.split()]
    except Exception as e:
        print(f"Error processing line: {line}")
        print(f"Error: {e}")
        return [], []

    # filter the line/session shorter than min_len, but only if min_len > 0
    if min_len > 0 and len(line) < min_len:
        return [], []

    # max seq len
    if seq_len is not None:
        line = line[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.array(line)

    # if time duration exists in data
    if line.shape[1] == 2:
        tim = line[:,1].astype(float)
        line = line[:, 0]

        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
    else:
        line = line.squeeze()
        # if time duration doesn't exist, then create a zero array for time
        tim = np.zeros(line.shape)

    logkey_seqs = []
    time_seq = []
    for i in range(0, len(line), window_size):
        logkey_seqs.append(line[i:i + window_size])
        time_seq.append(tim[i:i + window_size])

    return logkey_seqs, time_seq


def generate_train_valid(data_path, window_size=20, adaptive_window=True,
                         sample_ratio=1, valid_size=0.1, output_path=None,
                         scale=None, scale_path=None, seq_len=None, min_len=0):
    with open(data_path, 'r') as f:
        data_iter = f.readlines()

    num_session = int(len(data_iter) * sample_ratio)
    # only even number of samples, or drop_last=True in DataLoader API
    # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # num_session += num_session % 2

    test_size = int(min(num_session, len(data_iter)) * valid_size)
    # Ensure test_size is at least 1 if valid_size > 0
    if valid_size > 0 and test_size == 0:
        test_size = 1
    # only even number of samples
    # test_size += test_size % 2

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("="*40)

    logkey_seq_pairs = []
    time_seq_pairs = []
    session = 0
    for line in tqdm(data_iter):
        if session >= num_session:
            break
        session += 1

        logkeys, times = fixed_window(line, window_size, adaptive_window, seq_len, min_len)
        logkey_seq_pairs += logkeys
        time_seq_pairs += times

    logkey_seq_pairs = np.array(logkey_seq_pairs)
    time_seq_pairs = np.array(time_seq_pairs)

    # Check if we have any sequences
    if len(logkey_seq_pairs) == 0:
        print("=> Total available sequences: 0")
        # Create empty arrays with the right shape
        logkey_trainset = np.array([])
        logkey_validset = np.array([])
        time_trainset = np.array([])
        time_validset = np.array([])
    elif len(logkey_seq_pairs) == 1:
        # Special case: only one sequence, use it for both training and validation
        print("=> Only one sequence available, using it for both training and validation")
        logkey_trainset = logkey_seq_pairs
        logkey_validset = logkey_seq_pairs
        time_trainset = time_seq_pairs
        time_validset = time_seq_pairs
    else:
        # Adjust test_size if needed
        if test_size >= len(logkey_seq_pairs):
            test_size = max(1, len(logkey_seq_pairs) // 2)  # At least 1, at most half
            print(f"=> Adjusted validation size to {test_size}")
            
        logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(logkey_seq_pairs,
                                                                                          time_seq_pairs,
                                                                                          test_size=test_size,
                                                                                          random_state=1234)

    # sort seq_pairs by seq len if arrays are not empty
    if len(logkey_trainset) > 0:
        train_len = list(map(len, logkey_trainset))
        train_sort_index = np.argsort(-1 * np.array(train_len))
        logkey_trainset = logkey_trainset[train_sort_index]
        time_trainset = time_trainset[train_sort_index]
    
    if len(logkey_validset) > 0:
        valid_len = list(map(len, logkey_validset))
        valid_sort_index = np.argsort(-1 * np.array(valid_len))
        logkey_validset = logkey_validset[valid_sort_index]
        time_validset = time_validset[valid_sort_index]

    print("="*40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("="*40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset

