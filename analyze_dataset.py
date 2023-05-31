import re
from statistics import mean, median, stdev

# open annotations file
with open('challenge_anno.txt') as f:
    score_labels_txt = f.readlines()

# gather health severity ordinals
health_score = [int(re.search('\t(.*)\n', x).group(1)) for x in score_labels_txt[1:]]

# find mean and median
health_mean = mean(health_score)
health_median = median(health_score)
health_stdev = stdev(health_score)
print()