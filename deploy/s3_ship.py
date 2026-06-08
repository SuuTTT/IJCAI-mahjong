#!/usr/bin/env python3
"""
s3_ship.py — push deploy artifacts to S3 and mint presigned download links, or pull data back.
Auth: uses the EC2 instance IAM role automatically (no secrets), or ~/.aws creds, or env vars.

  python3 deploy/s3_ship.py up   <bucket> <dir-or-file> [--prefix ship] [--days 7]
  python3 deploy/s3_ship.py link <bucket> <key> [--days 7]          # presign an existing object
  python3 deploy/s3_ship.py pull <bucket> <key> <local>             # download (data from user)
  python3 deploy/s3_ship.py geturl <presigned_url> <local>          # wget a link the user sent
"""
import sys, os, argparse, hashlib, urllib.request

def _md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    u = sub.add_parser('up');   u.add_argument('bucket'); u.add_argument('path'); u.add_argument('--prefix', default='ship'); u.add_argument('--days', type=int, default=7)
    l = sub.add_parser('link'); l.add_argument('bucket'); l.add_argument('key'); l.add_argument('--days', type=int, default=7)
    p = sub.add_parser('pull'); p.add_argument('bucket'); p.add_argument('key'); p.add_argument('local')
    g = sub.add_parser('geturl'); g.add_argument('url'); g.add_argument('local')
    a = ap.parse_args()

    if a.cmd == 'geturl':                       # no boto3 needed — just fetch a presigned/public URL
        urllib.request.urlretrieve(a.url, a.local)
        print(f"{a.local}  md5={_md5(a.local)}")
        return

    import boto3
    s3 = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'ap-southeast-1'))

    def presign(bucket, key, days):
        return s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=days * 86400)

    if a.cmd == 'up':
        files = ([os.path.join(a.path, f) for f in os.listdir(a.path)] if os.path.isdir(a.path) else [a.path])
        files = [f for f in files if os.path.isfile(f)]
        print(f"uploading {len(files)} file(s) to s3://{a.bucket}/{a.prefix}/ (links valid {a.days}d):\n")
        for f in sorted(files):
            key = f"{a.prefix}/{os.path.basename(f)}"
            s3.upload_file(f, a.bucket, key)
            print(f"# {os.path.basename(f)}  md5={_md5(f)}")
            print(presign(a.bucket, key, a.days) + "\n")
    elif a.cmd == 'link':
        print(presign(a.bucket, a.key, a.days))
    elif a.cmd == 'pull':
        s3.download_file(a.bucket, a.key, a.local)
        print(f"{a.local}  md5={_md5(a.local)}")

if __name__ == '__main__':
    main()
