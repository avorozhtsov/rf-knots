# Reviewed Python child transport failure and separate v31a repair

First v31 supervisor resume session5858 failed BEFORE optimizer setup completed.
Pinned supervisor spawn explicitly replaces PYTHONPATH, dropping the opt-in
sitecustomize directory. RF flags alone do not load a reader. Child setup failed
FileNotFoundError on archived main-intake001563..001566/bank/path-00000.json.
This is a transport engineering failure, NOT missing/corrupt data, numerical
failure or disk guard. No native update or new full checkpoint occurred.
Original state exit1/runtime log and full2056741 SHA remain preserved.

Separate v31a launcher/sitecustomize/child-env bridge preserve the original pinned
v31 reader and ALL v30 model/source/spec pins. Only explicit opt-in Python Popen
children regain the sitecustomize directory even if a caller replaces PYTHONPATH.
No global Python installation changes. Child and grandchild probes must read the
exact archived payload under the pinned supervisor override before another launch.
Non-Python commands are unchanged. New bridge files have independent runtime pins.

The reviewed resume entrypoint permits ONLY this exact pre-optimizer transport
failure, full2056741/unchanged progress/ref and completed305..308. It preserves the
actual exit1 state and log in a review receipt; it supplies the original clean
parent exit0 eligibility only IN MEMORY for the pinned supervisor resume guard,
with an explicit review field. No broad force-resume, no numerical/input bypass,
no repeated searches, no rewritten original source. A one-shot launch receipt
prevents blind retry. Original full parent finite71AdamW/RNG/EMA/A/balance origin
and exact semantic snapshot are checked again. Future clean resumes use v31a
launcher directly and normal original safety predicates.

Storage remains v31 CAS, source window305..308 already passed12views/703episodes/
431.245805CPU/25newpaths; never rerun. Newdata awaits normal>=3600continuouswall
audited immutable handoff, not a manual shortcut. Native live>2056741 and new
recovery path intersection in AdamW must be separately proven. Preserve A1628687,
v3origin206004.2251009539/MAIN39168, balanceorigin1645639, actual50:50 component
norms and withinMCTS source50:50; zeroextraaux/rehearsal. No deletion or evaluation.
