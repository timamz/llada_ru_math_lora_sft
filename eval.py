import warnings
warnings.filterwarnings('ignore')

import re
import time
import inspect
from types import SimpleNamespace
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset

import torch
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

import dllm
import dllm.utils
from peft import PeftModel

from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r'^\\text\{(?P<text>.+?)\}$', answer)
        if m is not None:
            answer = m.group('text').strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace('\n', '')
    string = string.replace('\\!', '')
    string = string.replace('\\\\', '\\')

    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    string = string.replace('\\$', '')
    string = _remove_right_units(string)

    string = string.replace('\\%', '')
    string = string.replace('%', '')

    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    string = _fix_sqrt(string)
    string = string.replace(' ', '')
    string = _fix_fracs(string)

    if string == '0.5':
        string = '\\frac{1}{2}'

    string = _fix_a_slash_b(string)
    return string


BAD_SUBSTRINGS = ['^{', '^(']
BAD_REGEXES = [r'\^[0-9]+\^', r'\^[0-9][0-9]+']
TUPLE_CHARS = '()[]'


def _sympy_parse(expr: str):
    py_expr = expr.replace('^', '**')
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace('\\tfrac', '\\frac')
    expr = expr.replace('\\dfrac', '\\frac')
    expr = expr.replace('\\frac', ' \\frac')
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    expr = expr.replace('√', 'sqrt')
    expr = expr.replace('π', 'pi')
    expr = expr.replace('∞', 'inf')
    expr = expr.replace('∪', 'U')
    expr = expr.replace('·', '*')
    expr = expr.replace('×', '*')

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r'^-?[0-9]+.?/0*[1-9][0-9]*.?$', expr))


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r'(\d)(,)(\d\d\d)($|\D)')
    while True:
        next_expr = p1.sub(r'\1\3\4', expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(',', '')
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile(r'([0-9]) +([0-9])')
    return p1.sub(r'\1+\2', step)


def _normalize(expr: str) -> str:
    if expr is None:
        return None

    m = re.search(r'^\\text\{(?P<text>.+?)\}$', expr)
    if m is not None:
        expr = m.group('text')

    expr = expr.replace('\\%', '%')
    expr = expr.replace('\\$', '$')
    expr = expr.replace('$', '')
    expr = expr.replace('%', '')

    expr = expr.replace(' or ', ' , ')
    expr = expr.replace(' and ', ' , ')
    expr = expr.replace(' или ', ' , ')
    expr = expr.replace(' и ', ' , ')

    expr = expr.replace('million', '*10^6')
    expr = expr.replace('billion', '*10^9')
    expr = expr.replace('trillion', '*10^12')

    expr = re.sub(r'\bмиллион(а|ов)?\b', '*10^6', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bмиллиард(а|ов)?\b', '*10^9', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bтриллион(а|ов)?\b', '*10^12', expr, flags=re.IGNORECASE)

    for unit in [
        'degree',
        'cm',
        'centimeter',
        'meter',
        'mile',
        'second',
        'minute',
        'hour',
        'day',
        'week',
        'month',
        'year',
        'foot',
        'feet',
        'inch',
        'yard',
    ]:
        expr = re.sub(f'{unit}(es)?(s)? *(\\^[0-9]+)?', '', expr, flags=re.IGNORECASE)

    ru_units = [
        r'градус(ов|а)?',
        r'см',
        r'сантиметр(ов|а)?',
        r'метр(ов|а)?',
        r'миля(мили|миль)?',
        r'секунд(а|ы|)?',
        r'минута(ы|)?|минут',
        r'час(ов|а)?',
        r'день|дня|дней',
        r'недел(я|и|ь)|недель',
        r'месяц(ев|а)?',
        r'год|года|лет',
        r'фут(ов|а)?',
        r'дюйм(ов|а)?',
        r'ярд(ов|а)?',
    ]
    for pat in ru_units:
        expr = re.sub(rf'\b{pat}\b *(\\^[0-9]+)?', '', expr, flags=re.IGNORECASE)

    expr = re.sub(r'\^ *\\circ', '', expr)

    if len(expr) > 0 and expr[0] == '{' and expr[-1] == '}':
        expr = expr[1:-1]

    expr = re.sub(r',\\! *', '', expr)

    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))

    if '\\' in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    expr = re.sub(r'- *', '-', expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(' ', '')

    expr = expr.replace('{', '')
    expr = expr.replace('}', '')

    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace('sqrt', '')
    expr = expr.replace('frac', '')
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    try:
        expr = f'({ground_truth_normalized})-({given_normalized})'
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            return simplified == 0
    except Exception:
        pass
    return False


def split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(',')]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    if given_answer is None:
        return False

    gt_mathd = normalize_answer(ground_truth)
    pred_mathd = normalize_answer(given_answer)
    if gt_mathd == pred_mathd:
        return True

    gt = _normalize(ground_truth)
    pred = _normalize(given_answer)

    if gt is None:
        return False
    if gt == pred:
        return True
    if len(pred) == 0:
        return False

    gt_elems = split_tuple(gt)
    pred_elems = split_tuple(pred)

    if len(gt_elems) > 1 and (gt[0] != pred[0] or gt[-1] != pred[-1]):
        return False
    if len(gt_elems) != len(pred_elems):
        return False

    for gt_elem, pred_elem in zip(gt_elems, pred_elems):
        if _is_frac(gt_elem) and _is_frac(pred_elem):
            ok = (gt_elem == pred_elem)
        elif _str_is_int(gt_elem) != _str_is_int(pred_elem):
            ok = False
        else:
            ok = are_equal_under_sympy(gt_elem, pred_elem)
        if not ok:
            return False

    return True


def build_prompt_ru(problem: str) -> str:
    return (
        'Дайте четкий и лаконичный ответ на следующий математический вопрос '
        'в формате LaTeX и представьте окончательный ответ в виде '
        '\\(\\boxed{x}\\), где X — полностью упрощенное решение.\n\n'
        'Пример:\n'
        'Задача: \\(\\int_0^1 (3x^2 + 2x) \\,dx\\)\n'
        'Решение: '
        '\\(\\int (3x^2 + 2x) \\,dx = x^3 + x^2 + C\\) '
        'Вычисление от 0 до 1: '
        '\\((1^3 + 1^2) - (0^3 + 0^2) = 1 + 1 - 0 = 2\\) '
        '\\(\\boxed{2}\\)\n'
        'Окончательный ответ: \\(\\boxed{2}\\)\n\n'
        'Теперь решите следующую задачу:\n'
        f'Задача: {problem}\n'
        'Решение:'
    )


def extract_boxed(text: str) -> str:
    i = text.rfind(r'\boxed{')
    if i == -1:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else ''

    j = i + len(r'\boxed{')
    depth = 1
    out = []
    while j < len(text) and depth > 0:
        ch = text[j]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        j += 1

    return ''.join(out).strip()


def load_llada_base(model_name_or_path: str):
    script_args = SimpleNamespace(
        model_name_or_path=model_name_or_path,
        trust_remote_code=True,
        load_in_4bit=False,
        lora=False,
    )
    tok = dllm.utils.get_tokenizer(model_args=script_args)
    model = dllm.utils.get_model(model_args=script_args).eval()
    return model, tok


def load_llada_lora(base_model_name_or_path: str, adapter_path: str):
    script_args = SimpleNamespace(
        model_name_or_path=base_model_name_or_path,
        trust_remote_code=True,
        load_in_4bit=False,
        lora=False,
    )
    tok = dllm.utils.get_tokenizer(model_args=script_args)
    base_model = dllm.utils.get_model(model_args=script_args)
    model = PeftModel.from_pretrained(base_model, adapter_path).eval()
    return model, tok


def _sampler_sample(sampler, inputs_t, cfg: MDLMSamplerConfig):
    sig = inspect.signature(sampler.sample)
    kwargs = {'return_dict': True}
    if 'config' in sig.parameters:
        kwargs['config'] = cfg
        return sampler.sample(inputs_t, **kwargs)

    if hasattr(sampler, 'config'):
        sampler.config = cfg
    elif hasattr(sampler, 'cfg'):
        sampler.cfg = cfg
    return sampler.sample(inputs_t, return_dict=True)


@torch.inference_mode()
def llada_generate(model, tok, prompt: str, cfg: MDLMSamplerConfig) -> str:
    sampler = MDLMSampler(model=model, tokenizer=tok)

    messages = [
        [{'role': 'user', 'content': prompt}],
    ]

    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )

    if isinstance(inputs, list):
        inputs_t = torch.tensor(inputs, device=model.device)
    else:
        inputs_t = inputs.to(model.device)

    outputs = _sampler_sample(sampler, inputs_t, cfg)

    sequences = dllm.utils.decode_trim(
        tok,
        outputs.sequences.tolist(),
        inputs_t.tolist() if hasattr(inputs_t, 'tolist') else inputs,
    )

    return sequences[0]


def eval_one_run(spec, df, limit=None):
    if spec['kind'] == 'base':
        model, tok = load_llada_base(spec['model_path'])
    else:
        model, tok = load_llada_lora(spec['model_path'], spec['lora_path'])

    cfg = MDLMSamplerConfig(
        max_new_tokens=spec['gen']['max_new_tokens'],
        max_length=spec['gen'].get('max_length', None),
        block_size=spec['gen']['block_size'],
        steps=spec['gen']['steps'],
        temperature=spec['gen']['temperature'],
        remasking=spec['gen']['remasking'],
        stochastic_transfer=spec['gen']['stochastic_transfer'],
        cfg_scale=spec['gen']['cfg_scale'],
        cfg_keep_tokens=spec['gen'].get('cfg_keep_tokens', None),
        suppress_tokens=spec['gen'].get('suppress_tokens', None),
        begin_suppress_tokens=spec['gen'].get('begin_suppress_tokens', None),
        right_shift_logits=spec['gen']['right_shift_logits'],
    )

    rows = []
    correct = 0
    n = 0
    t0 = time.time()

    if limit:
        it = df.head(limit).itertuples(index=False)
        total = limit
    else:
        it = df.itertuples(index=False)
        total = len(df)

    pbar = tqdm(it, total=total, desc=spec['run_name'], leave=False)
    for ex in pbar:
        problem = ex.problem
        gt = ex.answer
        prompt = build_prompt_ru(problem)

        try:
            raw = llada_generate(model, tok, prompt, cfg)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raw = ''
        except Exception:
            raw = ''

        pred = extract_boxed(raw)
        ok = grade_answer(pred, gt)

        correct += int(ok)
        n += 1

        if n % 10 == 0:
            pbar.set_postfix({
                'acc': round(correct / n, 4),
                'boxed': round(sum(1 for r in rows if r['pred_extracted']) / max(len(rows), 1), 4),
            })

        rows.append({
            'run': spec['run_name'],
            'run_description': spec['run_description'],
            'model': spec['name'],
            'model_path': spec['model_path'],
            'kind': spec['kind'],
            'lora_path': spec.get('lora_path', ''),

            'config': spec['gen']['name'],
            'steps': spec['gen']['steps'],
            'block_size': spec['gen']['block_size'],
            'max_new_tokens': spec['gen']['max_new_tokens'],
            'temperature': spec['gen']['temperature'],
            'remasking': spec['gen']['remasking'],
            'stochastic_transfer': spec['gen']['stochastic_transfer'],
            'cfg_scale': spec['gen']['cfg_scale'],
            'right_shift_logits': spec['gen']['right_shift_logits'],

            'unique_id': ex.unique_id,
            'subject': ex.subject,
            'level': ex.level,
            'ground_truth': gt,

            'pred_extracted': pred,
            'is_correct': bool(ok),
            'raw_output': raw,
        })

        del raw
        torch.cuda.empty_cache()

    sec = time.time() - t0
    result_df = pd.DataFrame(rows)
    summary = {
        'run': spec['run_name'],
        'run_description': spec['run_description'],
        'model': spec['name'],
        'kind': spec['kind'],
        'config': spec['gen']['name'],
        'steps': spec['gen']['steps'],
        'block_size': spec['gen']['block_size'],

        'n': n,
        'accuracy': correct / max(n, 1),
        'boxed_rate': (result_df['pred_extracted'].astype(str).str.len() > 0).mean() if n else 0.0,
        'seconds_total': sec,
        'seconds_per_task': sec / max(n, 1),
    }
    return summary, result_df


def main():
    # ---- Config ----
    LLADA_BASE = 'GSAI-ML/LLaDA-8B-Instruct'
    LORA_PATH = 'checkpoint-329'

    LIMIT = None  # set e.g. 50 for quick test

    BLOCK_SIZE = 512
    STEPS_LIST = [150, 100, 50]

    ds = load_dataset('AvitoTech/ru_math500', split='test')
    df = ds.to_pandas()

    models = [
        {
            'name': 'llada-8b-base',
            'kind': 'base',
            'model_path': LLADA_BASE,
            'description': 'Base weights (no LoRA)',
        },
        {
            'name': 'llada-8b-lora',
            'kind': 'lora',
            'model_path': LLADA_BASE,
            'lora_path': LORA_PATH,
            'description': 'Base + LoRA adapter',
        },
    ]

    gen_configs = []
    for steps in STEPS_LIST:
        gen_configs.append({
            'name': f'steps{steps}_bs{BLOCK_SIZE}',
            'max_new_tokens': 512,
            'max_length': None,
            'block_size': BLOCK_SIZE,
            'steps': steps,
            'temperature': 0.0,
            'remasking': 'low_confidence',
            'stochastic_transfer': False,
            'cfg_scale': 0.0,
            'cfg_keep_tokens': None,
            'suppress_tokens': None,
            'begin_suppress_tokens': None,
            'right_shift_logits': False,
        })

    runs = []
    for m in models:
        for g in gen_configs:
            runs.append({
                **m,
                'gen': g,
                'run_name': f'{m["name"]}__{g["name"]}',
                'run_description': f'{m["description"]} | {g["name"]}',
            })

    all_summaries = []
    all_runs = []

    for spec in tqdm(runs, desc='runs', total=len(runs)):
        summary, run_df = eval_one_run(spec, df, limit=LIMIT)
        all_summaries.append(summary)
        all_runs.append(run_df)

    summary_df = pd.DataFrame(all_summaries)
    runs_df = pd.concat(all_runs, ignore_index=True)

    cols = [
        'run', 'model', 'kind', 'config',
        'accuracy', 'boxed_rate', 'seconds_per_task', 'n',
        'steps', 'block_size',
    ]
    print(summary_df[cols].sort_values(['accuracy', 'boxed_rate'], ascending=False).to_string(index=False))

    by_subject = (
        runs_df.groupby(['run', 'subject'], as_index=False)
        .agg(
            n=('is_correct', 'size'),
            accuracy=('is_correct', 'mean'),
            boxed_rate=('pred_extracted', lambda s: (s.astype(str).str.len() > 0).mean()),
            steps=('steps', 'first'),
            block_size=('block_size', 'first'),
            kind=('kind', 'first'),
        )
        .sort_values(['run', 'accuracy', 'n'], ascending=[True, False, False])
    )

    by_level = (
        runs_df.groupby(['run', 'level'], as_index=False)
        .agg(
            n=('is_correct', 'size'),
            accuracy=('is_correct', 'mean'),
            boxed_rate=('pred_extracted', lambda s: (s.astype(str).str.len() > 0).mean()),
            steps=('steps', 'first'),
            block_size=('block_size', 'first'),
            kind=('kind', 'first'),
        )
        .sort_values(['run', 'level'])
    )

    summary_df.to_csv('eval_res/ru_math500_llada_summary.csv', index=False)
    runs_df.to_csv('eval_res/ru_math500_llada_runs.csv', index=False)
    by_subject.to_csv('eval_res/ru_math500_llada_by_subject.csv', index=False)
    by_level.to_csv('eval_res/ru_math500_llada_by_level.csv', index=False)

    print('\nSaved:')
    print(' - ru_math500_llada_summary.csv')
    print(' - ru_math500_llada_runs.csv')
    print(' - ru_math500_llada_by_subject.csv')
    print(' - ru_math500_llada_by_level.csv')


if __name__ == '__main__':
    main()
