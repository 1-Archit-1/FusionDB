def calculate_recall(found_indices, ground_truth_indices):
    # Convert to sets for efficient intersection
    found_set = set(found_indices)
    truth_set = set(ground_truth_indices)
    
    k = len(truth_set)
    if k == 0:
        # If ground truth is empty, recall is 1.0 (100%) if found is also empty
        # or 0.0 if found is not empty. Or just define as 1.0.
        recall = 1.0 if len(found_set) == 0 else 0.0
    else:
        # Calculate the number of relevant items found
        relevant_found = len(found_set.intersection(truth_set))
        recall = relevant_found / k
    
    return recall