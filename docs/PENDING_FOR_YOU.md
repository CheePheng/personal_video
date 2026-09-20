# Open items

Nothing here blocks you. The app works today.

## 1. Cloudflare email — RESOLVED
Confirmed `doctorwilddoctorwild@gmail.com` (the `.cm` was a typo).

## 2. Permanent URL — DECIDED: staying on Quick Tunnel

You have one domain on the account (`cctgroup.my`, Free plan) and chose not to
use it for this. That is the only domain Cloudflare would let a named tunnel
attach to, so the named-tunnel path is closed by choice.

`cloudflared tunnel login` was cancelled. **No `cert.pem` was written and
nothing is linked to your Cloudflare account.**

We run on **Quick Tunnel** instead:

- free forever, no account, no card
- URL is a random `*.trycloudflare.com` hostname with nothing tying it to you
  or to `cctgroup.my`
- `start.bat` prints the current link each launch
- the link changes on every restart — this is the only downside

### If you ever want a permanent link

Two ways, both requiring a decision you have not made:

1. **Random subdomain of your existing domain** — free, permanent, but
   `cctgroup.my` appears in the URL:

       cloudflared tunnel login
       cloudflared tunnel create faceswap
       cloudflared tunnel route dns faceswap k3m9x2p.cctgroup.my

2. **Register a separate throwaway domain** (~$10/yr, needs your payment
   details) and do the same against that zone.

I cannot create a domain for you — domains must be registered and paid for.

## 3. Git — not committed
Nothing has been committed; you did not ask. Say the word and I will.
