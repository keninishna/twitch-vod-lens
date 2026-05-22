"""Check 758s analysis from live pipeline run."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'}, timeout=20, encoding='utf-8')
idx = out.find('{')
d = json.loads(out[idx:])

print("=== 758s CLIP ANALYSIS ===")
for c in d['clip_details']:
    if c.get('start') == 758:
        a = c['analysis']
        print(json.dumps({k: a[k] for k in ('clip_worthiness','narrative_type','has_narrative_payoff','clip_point','trigger','payoff') if k in a}, indent=2))
        err = a.get('error','')
        print(f"error: {err[:50] if err else '(none)'}")
        break

print("\n=== ALL CLIP RAW SCORES ===")
for c in d['clip_details']:
    a = c['analysis']
    s = c.get('start')
    cw = a.get('clip_worthiness', a.get('clip_worthy', '?'))
    nt = a.get('narrative_type', '?')
    err = a.get('error', '')
    print(f"  {s}s: {cw}/10 type={nt} {'⚠️ '+err[:30] if err else ''}")

print("\n=== SELECTED CLIPS with INTELLIGENCE ===")
for c in d['final_ranking']['final_selected_clips']:
    ir = c.get('intelligence_report',{})
    print(f"  start={c['start']} score={c['score']}")
    print(f"    title: {c.get('clip_point','')[:60]}")
    print(f"    why: {ir.get('why_selected','')[:80]}")
    print(f"    scores: {c.get('platform_scores',{})}")
    print(f"    recs: {c.get('platform_recommendations',[])}")