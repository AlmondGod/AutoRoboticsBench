# RoboCasa OpenDrawer Task-Index Conditioning

Date: 2026-06-11

Setup:
- Dataset: RoboCasa OpenDrawer LeRobot dataset
- Conditioning: `observation -> task_index embedding -> action chunk`
- Task ids: RoboCasa `task_index` 0 and 1
- Policy: temporal chunk BC, chunk horizon 16, commit 16
- Held-out task 0 episodes: 87, 92, 93, 94, 98, 100, 101
- Held-out task 1 episodes: 88, 89, 90, 91, 95, 96, 97, 99

Results:

| run | train data | held-out eval | success |
| --- | --- | --- | --- |
| shared task-index-conditioned policy | task 0 + task 1, conditioned on task_index | 15 combined episodes | 2/15 |
| shared policy, task 0 subset | task 0 + task 1, conditioned on task_index | 7 task-0 episodes | 1/7 |
| shared policy, task 1 subset | task 0 + task 1, conditioned on task_index | 8 task-1 episodes | 1/8 |
| task-1 specialist | task 1 only | 8 task-1 episodes | 1/8 |

Takeaway:
- Simple task-index conditioning did not improve over the task-0 specialist/ensemble.
- Task index 1 is much harder under the current chunk-BC setup: both shared conditioning and a task-1 specialist reached only 1/8 held-out success.
- The current best overall task-0 model remains the 50/50 ensemble of two chunk-16 BC policies at 5/7 on task index 0.
