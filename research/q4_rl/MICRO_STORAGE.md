# Linear episode storage

The frozen first macro/micro pilot remains on release `f4162e4a`. Its first micro batch took 206.98 wall seconds, with 25.74 seconds updating the network and 48.22 summed worker CPU seconds across eight workers. The driver also serialized every previous episode after each return: 136 episode serializations for a 16-episode batch.

The next revision writes each episode once, then rewrites only a small hash index: 16 episode serializations plus 16 tiny indexes. It preserves decoded worker objects, failures, attempt identities and checkpoint transactions. No feature, reward, action, optimizer or statistical definition changes. `read_batch` supports old gzip lists and new indexes, verifying episode hashes. Files remain in the existing object-store include patterns. A crash between episode write and index publication can leave an identifiable extra episode beside the last valid index; that file is retained.

Validation: 14 journal and micro-driver tests passed, including partial worker failure, interrupted-update replay, matching uninterrupted final weights, stable prior episodes, corruption rejection and legacy/new decoded equality. This is an operation-count improvement; no policy efficacy or measured end-to-end speedup is claimed yet. The next PPO and attention comparison must share this revision.
