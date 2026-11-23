"""Local scalar, dimension-aware measurement, and four-choice reward verifiers."""
import math,re
from numerical_protocol import numerical_score,extract_boxed_answer
from task3_measurement_metrics import parse_measurement,component_errors
from choice import extract_choice

def local_score(completion,kind,reference,gold='',finish_reason=None,is_truncated=False):
    answer,found=extract_boxed_answer(completion)
    result={'response':answer,'boxed_answer_found':found,'reward':0.0,'reward_type':kind}
    if not found or finish_reason not in (None,'','stop') or is_truncated:return result
    if kind=='rex_scalar':return dict(numerical_score(completion,reference,finish_reason,is_truncated),reward_type=kind)
    if kind=='choice':
        # Match the official A-D verifier, but never reward a list of several choices.
        choices={x.upper() for x in re.findall(r'\b([A-D])\b',answer,re.IGNORECASE)}
        prediction=extract_choice(answer)
        result.update(predicted_choice=prediction,reward=float(len(choices)<=1 and prediction==gold))
        return result
    if kind!='3d_measurement':raise ValueError(f'Unknown local verifier {kind}')
    if re.search(r'(?<![\w.])-\s*\d',answer):return result
    ref,pred=parse_measurement(reference),parse_measurement(answer)
    if not ref or not pred or ref['kind']!=pred['kind'] or len(ref['values'])!=len(pred['values']):return result
    if pred['interval'] or pred['inequality'] is not None:return result
    if any(not math.isfinite(x) or x<0 for x in pred['values']):return result
    errors,_=component_errors(ref,pred)
    scales=[max(5.0,0.2*abs(x)) for x in sorted(ref['values'],reverse=True)]
    result.update(component_errors=errors,reward=math.exp(-sum(e/s for e,s in zip(errors,scales))/len(errors)))
    return result
