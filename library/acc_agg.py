from collections import defaultdict

def average(teachers_attachments, weights=None):
    K = len(teachers_attachments)
    n = len(teachers_attachments[0])
    if weights is None:
        weights = [1]*K
    attachment_scores = [[sum([(t[dependent]==head)*w for t, w in zip(teachers_attachments, weights)]) for head in range(n+1)] for dependent in range(n)]

    records = [[[(-1, None) for head in range(l)] for b in range(n-l+1)] for l in range(1, n+1)]

    for b in range(n):
        records[0][b][0] = (0, [0])

    for l in range(2, n+1): # length of span
        for b in range(n-l+1): # begin of span
            e = b+l
            for left_size in range(1, l):
                right_size = l-left_size
                for left_h in range(left_size):
                    for right_h in range(right_size):
                        left_score, left_attachment = records[left_size-1][b][left_h]
                        right_score, right_attachment = records[right_size-1][b+left_size][right_h]
                        inner_score, concat_attachment = left_score+right_score, left_attachment+right_attachment
                        left_to_right_score = attachment_scores[b+left_h][b+left_size+right_h+1]
                        right_to_left_score = attachment_scores[b+left_size+right_h][b+left_h+1]
                        if inner_score+left_to_right_score > records[l-1][b][left_size+right_h][0]:
                            new_attachment = concat_attachment.copy()
                            new_attachment[left_h] = b+left_size+right_h+1
                            records[l-1][b][left_size+right_h] = (inner_score+left_to_right_score, new_attachment)
                        if inner_score+right_to_left_score > records[l-1][b][left_h][0]:
                            new_attachment = concat_attachment.copy()
                            new_attachment[left_size+right_h] =  b+left_h+1
                            records[l-1][b][left_h] = (inner_score+right_to_left_score, new_attachment)
    records = records[-1][0]
    for i in range(n):
        score, attachment = records[i]
        records[i] = (score+attachment_scores[i][0], attachment)
    
    _, answer = max(records, key=lambda x: x[0])
    return answer