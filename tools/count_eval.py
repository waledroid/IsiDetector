"""Count-accuracy harness: run a clip through /api/start, compare the resulting
per-class counts to a hand-labeled truth, report miss rate (under-count).

Usage:
  python tools/count_eval.py --base http://localhost:9501 --password change-me \
      --source /opt/isitec/webapp/isitec_api/uploads/testvid.mp4 \
      --truth '{"carton": 20, "polybag": 7}' --seconds 90
"""
import argparse, json, time, urllib.request


def _post(base, path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('X-Dev-Token', token)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read() or b'{}')


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=15) as r:
        return json.loads(r.read() or b'{}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://localhost:9501')
    ap.add_argument('--password', default='change-me')
    ap.add_argument('--source', required=True)
    ap.add_argument('--truth', required=True, help='JSON dict of true counts per class')
    ap.add_argument('--seconds', type=int, default=90)
    ap.add_argument('--model', default='yolo')
    args = ap.parse_args()

    truth = json.loads(args.truth)
    token = _post(args.base, '/api/dev-auth', {'password': args.password}).get('token')
    _post(args.base, '/api/start', {'source': args.source, 'model_type': args.model}, token)
    try:
        deadline = time.monotonic() + args.seconds
        last = {}
        while time.monotonic() < deadline:
            st = _get(args.base, '/api/stats')
            last = st.get('counts', {})
            if not st.get('is_running', True):
                break
            time.sleep(2)
    finally:
        _post(args.base, '/api/stop', {}, token)

    print('class      truth  counted  missed  miss%')
    total_t = total_c = 0
    for cls, t in truth.items():
        c = int(last.get(cls, 0)); m = t - c
        total_t += t; total_c += c
        print(f'{cls:<10} {t:>5} {c:>8} {m:>7} {(100*m/t if t else 0):>6.1f}')
    miss = total_t - total_c
    print(f'{"TOTAL":<10} {total_t:>5} {total_c:>8} {miss:>7} {(100*miss/total_t if total_t else 0):>6.1f}')


if __name__ == '__main__':
    main()
