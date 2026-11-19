"""ID selective-classification risk at attainable confidence thresholds."""
import math

def risk_coverage(confidences, errors):
    if len(confidences)!=len(errors) or not confidences:
        raise ValueError('Need equally sized, nonempty confidence/error vectors.')
    if not all(math.isfinite(c) and 0<=c<=1 for c in confidences):
        raise ValueError('Confidence must be finite and in [0,1].')
    if not all(type(e) in (int,bool) and e in (0,1) for e in errors):
        raise ValueError('Errors must be binary.')
    groups={}
    for c,e in zip(confidences,errors):
        n,bad=groups.get(c,(0,0)); groups[c]=(n+1,bad+e)
    n_total=len(errors); kept=bad=0; previous=aurc=0.0; points=[]
    correct_total=n_total-sum(errors)
    for threshold,(n,wrong) in sorted(groups.items(),reverse=True):
        kept+=n; bad+=wrong
        coverage=kept/n_total; risk=bad/kept
        aurc+=(coverage-previous)*risk; previous=coverage
        points.append(dict(threshold=threshold,accepted=kept,coverage=coverage,errors=bad,risk=risk,
                           oracle_risk=max(0,kept-correct_total)/kept))
    return dict(n=n_total,total_errors=sum(errors),full_coverage_risk=sum(errors)/n_total,
                aurc_right_endpoint=aurc,points=points,
                convention='Equal-confidence samples accepted together. AURC is right-endpoint risk weighted by coverage increments; zero-coverage risk is undefined.',
                task='In-distribution selective classification, not OOD detection')
