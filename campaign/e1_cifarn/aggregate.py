#!/usr/bin/env python3
"""Merge per-noise-level E1 results (CIFAR-10N + CIFAR-100N) into
results/CIFARN_THRESHOLD.json with a factual verdict on the threshold
signature (gap vs noise rate) and the mechanism decomposition
(distill-alone vs ensemble-on-top). Tolerates missing levels (in-flight)."""
import json, os, time

resdir = 'results'
SECTIONS = {
    'c10': ['clean', 'aggre', 'rand1', 'rand2', 'rand3', 'worst'],
    'c100': ['c100_clean', 'c100_noisy'],
}


def row_of(d, lv):
    return {
        'noise': lv,
        'measured_noise_rate': d['measured_noise_rate'],
        'label_source': d['label_source'],
        'single_teacher': d['single_teacher'],
        'single_teacher_std': d['single_teacher_std'],
        'teacher_ens3': d['teacher_ens3'],
        'teacher_ens3_spread': d['teacher_ens3_spread'],
        'teacher_ens6': d['teacher_ens6'],
        'single_student': d['single_student'],
        'single_student_std': d['single_student_std'],
        'student_ens3': d['student_ens3'],
        'gap': d['gap_student_ens3_minus_teacher_ens3'],
        'distill_alone': d['single_student'] - d['single_teacher'],
        'teacher_ens_lift': d['teacher_ens3'] - d['single_teacher'],
        'student_ens_lift': d['student_ens3'] - d['single_student'],
    }


tables, missing, verdict_bits = {}, [], []
for sec, levels in SECTIONS.items():
    rows = []
    for lv in levels:
        p = os.path.join(resdir, f'e1_{lv}.json')
        if os.path.exists(p):
            with open(p) as f:
                rows.append(row_of(json.load(f), lv))
        else:
            missing.append(lv)
    tables[sec] = rows

c10 = tables['c10']
if len(c10) == 4:
    gaps = [r['gap'] for r in c10]
    rates = [r['measured_noise_rate'] for r in c10]
    mono = all(gaps[i + 1] > gaps[i] for i in range(3))
    verdict_bits.append(
        f"C10: gaps at noise {['%.3f' % x for x in rates]} = "
        f"{['%+.4f' % g for g in gaps]}; strictly monotonic: {mono}; "
        f"worst>clean: {gaps[-1] > gaps[0]}.")
    da = [r['distill_alone'] for r in c10]
    frac = (da[-1] / gaps[-1]) if abs(gaps[-1]) > 1e-9 else float('nan')
    verdict_bits.append(
        f"C10 mechanism: distill-alone = {['%+.4f' % x for x in da]}; "
        f"fraction of gap at worst = {frac:.2f}.")
else:
    verdict_bits.append('C10 INCOMPLETE.')

c100 = tables['c100']
if len(c100) == 2:
    g_cl, g_nz = c100[0]['gap'], c100[1]['gap']
    verdict_bits.append(
        f"C100: gap clean {g_cl:+.4f} -> noisy({c100[1]['measured_noise_rate']:.3f}) "
        f"{g_nz:+.4f}; noisy>clean: {g_nz > g_cl}. "
        f"C100 mechanism: distill-alone clean {c100[0]['distill_alone']:+.4f}, "
        f"noisy {c100[1]['distill_alone']:+.4f}.")
else:
    verdict_bits.append('C100 INCOMPLETE.')

out = {
    'experiment': 'E1 CIFAR-10N + CIFAR-100N real-label-noise: distill-then-ensemble noise-threshold (PLAN_domain_extension.md)',
    'date': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
    'design': '6 teachers/level; teacher_ens3 = avg of 2 disjoint trios (composition-averaged); '
              '3 KD students (alpha=0.7 soft targets = 6-teacher mean softmax, 0.3 CE(noisy,ls=0.1)); '
              'clean test acc; identical recipe all conditions; ResNet-18 (10- or 100-way head).',
    'table': tables['c10'],
    'c100_table': tables['c100'],
    'missing': missing,
    'verdict': ' '.join(verdict_bits),
}
with open(os.path.join(resdir, 'CIFARN_THRESHOLD.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
