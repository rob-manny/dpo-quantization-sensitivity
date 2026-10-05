# assets/

Drop the reward-margin plot here as `reward_margin.png` — it's referenced
from the main README ("Findings" section). Generate it from
`results/dpo_train_log.json`, e.g.:

```python
import json
import matplotlib.pyplot as plt

log = json.load(open('../results/dpo_train_log.json'))
steps = [e['step'] for e in log if 'rewards/margins' in e]
marg  = [e['rewards/margins'] for e in log if 'rewards/margins' in e]
plt.plot(steps, marg)
plt.xlabel('step'); plt.ylabel('rewards/margins')
plt.title('DPO reward margin (rising = learning the preference)')
plt.savefig('assets/reward_margin.png', dpi=150)
```
