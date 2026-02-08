chars ='កខគកខ'
# sorted(list(set(chars)))
print(f"set(chars): {set(chars)}")
print("\n")
print(f"list(set(chars)) {list(set(chars))}")
print("\n")
print(f"sorted {sorted(list(set(chars)))}")
sorted_chars=sorted(list(set(chars)))
char_to_id = {c: i + 1 for i, c in enumerate(sorted_chars)}
id_to_char = {i + 1: c for i, c in enumerate(sorted_chars)}

print(f"char_to_id {char_to_id}")
print(f"id_to_char {id_to_char}")