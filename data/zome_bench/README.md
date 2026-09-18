---
license: afl-3.0
---

# ZoME-Bench 

## data format
- "editing_prompt": the edited caption
- "original_prompt": the origin caption
- "blended_word": the subject to be edited
- "emphasize": the word which should be emphasized
- "ytid": id in YouTube
- "editing_type_id": described in the table following
- "editing_instruction": description of editing
- "neg_prompt": the prompt should be deleted ,only for type "delete instrument"  

**Editing Type Table**

| editing_type_id | Editing type                        |
| :-------------: | :---------------------------------: |
| 0               | Change instrument                   |
| 1               | Add instrument                      |
| 2               | Delete instrument                   |
| 3               | Change genre                        |
| 4               | Change mood                         |
| 5               | Change rhythm                       |
| 6               | Change background                   |
| 7               | Change melody                       |
| 8               | Extract instrument                  |
| 9               | Random picked from the first 9 type |

