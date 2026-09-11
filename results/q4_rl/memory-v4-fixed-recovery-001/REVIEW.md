# Fixed C panel continuation

This is a continuation of the interrupted original development panel, not a new
panel or a selected subset. The user approved the exact historical training
payload cleanup; that cleanup completed separately and did not remove these
evaluation records, models or source.

The recovery manifest binds 6656 expected records: 2581 retained and 4075 missing.
Independent local metadata review verified disjoint sets, no duplicates, the
original missing-record order, 13 arms per request, 62 frozen source hashes and
10 model bindings. The remote preparation decoded and validated all retained
records; it found zero failed episodes and zero damaged records. Failed optical
clear attempts are a distinct metric and are not asserted to be zero. Valid
failed episodes would also be retained by the same mechanism.

The original Linux environment remains mandatory. Reconstructing the cases on
Windows produced different ground-truth byte hashes in 219 of the 512 cases,
despite matching request identities and source files. This is not permission to
relax the frozen case-hash check: execution must reproduce the original matrix
on the original server. No Windows-generated cases replace server cases.

New recovery artifacts are capped at 1,000,000,000 bytes, including 8 MiB reserved
for outer logs and a 4096-byte administrative receipt. The separate supervisor
free-space guard is 18 GiB for this bounded evaluation. Four GAE training jobs
remain stopped. The original 22:19:58.910479 UTC deadline is unchanged.

The supervisor started at 2026-09-11 20:28:44 UTC, PID 1752515, with child/session
1752516. Initial actual affinity was CPUs 0–41 within the shared 50-CPU task pool;
the visible cgroup quota was 96 CPUs with ancestor limits outside the mount
unobservable. The first disk snapshot was 21,292,122,112 bytes free. Initial child
work is source/model/record revalidation; ten one-thread evaluation workers start
only after it passes. Subsequent process inspection verified all ten single-thread
workers (PIDs 1753445–1753454) in child process group 1752516. The immutable
execution-start receipt was uploaded and read back through the approved object
store, SHA256 `772fbc42e44d83d390342bf485ca9dde36fff78cea17956eb02838cd34533bfe`.
This establishes that the Linux matrix and retained-byte revalidation passed.
Object synchronization is enabled with the required S3 flag. This launch record
does not establish that the panel has completed.

The common references, actual billed time, failure retention, lower-bound
definition and 5000 seed-cluster bootstrap are unchanged. No new recommendation
may be based on an incomplete panel. Fresh128 remains separate and unstarted.
