import matplotlib.pyplot as plt
import json

"""
get distribution of: 
    1. length of the samples' text
    2. number of the sample's field
"""

with open('../data/train_data.json', 'r') as f:
    wiki_data = json.load(f)


def count_field(c_field):
    """count number of non-repetitive fields"""
    f_list = set()
    for c_f in c_field:
        f_list.add(c_f)
    return len(f_list)


average_len, average_field, average_no_repeat_field = 0, 0, 0
max_len, max_field = -1, -1
text_len_dict = {idx: 0 for idx in range(0, 270)}
field_cnt_dict = {idx: 0 for idx in range(0, 800)}

for sample in wiki_data:
    sample_len = len(sample['text'].split())
    field_cnt_no_repeat = count_field(sample['info_box']['field'])
    field_cnt = len(sample['info_box']['field'])
    average_len += sample_len
    average_field += field_cnt
    average_no_repeat_field += field_cnt_no_repeat
    if sample_len > max_len:
        max_len = sample_len
    if field_cnt > max_field:
        max_field = field_cnt
    if sample_len in text_len_dict:
        text_len_dict[sample_len] += 1
    if field_cnt in field_cnt_dict:
        field_cnt_dict[field_cnt] += 1

average_len /= len(wiki_data)
average_field /= len(wiki_data)
average_no_repeat_field /= len(wiki_data)

print(f'max_len of text: {max_len}')  # 269
print(f'average_len of text: {average_len:.2f}')  # 25.75
print(f'max_cnt of field: {max_field}')  # 791
print(f'average_cnt of field: {average_field:.2f}')  # 45.65
# 12.43
print(f'average_cnt of field (no-repeat): {average_no_repeat_field:.2f}')

# get distribution of the length of summary
plt.bar(list(text_len_dict.keys()), text_len_dict.values())
plt.xticks([idx for idx in range(0, 270, 30)])
plt.title('distribution of summary length')
plt.show()

# get distribution of the number of fields
plt.bar(list(field_cnt_dict.keys()), field_cnt_dict.values())
plt.xticks([idx for idx in range(0, 800, 100)])
plt.title('distribution of field number')
plt.show()
