Index 183: policy completed normally, 10 sources cleared, exit accepted.
Official completion was verified and jlog copied, but Add-Content could not open
completed.jsonl because a concurrent reader held a Windows sharing lock.
No ledger row was appended. Old worker absent at recovery; result and official
completed page still present. Preserve the original case and commit it once.
New Append-Ledger retries only file-open sharing violations for at most 30 sec;
it never retries writes, avoiding duplicate append after an uncertain write.
Recovery may use original verified completion-modal OCR plus the live completed
Q3 page when the modal has already been closed. No policy is rerun.
