"""Causal event replay and research training for the supplied Daegu export."""
from pathlib import Path
import argparse
import json
import hashlib
import zipfile
import numpy as np
import pandas as pd

PROFILES={'ALL':None,'AC_SLOW':['02'],'DC_COMBO':['04','05','06','08','10']}
CORE=['totalCount','availableCount','chargingCount','otherCount','unknownCount','hourSin','hourCos','dowSin','dowCos','profileIndex']
HISTORY=['availableRatio','chargingRatio','availableLag5','availableLag15','availableLag30','chargingLag15','availableMean30','availableMin30','availableTrend30','observedChanges30m','changesPerCharger30m','chargingElapsedMeanMin','chargingElapsedMaxMin','chargingElapsedValidRatio','meanOutputKw','acSlowFraction','dcComboFraction']

def save_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')

def parse_ts(x): return pd.to_datetime(x,format='%Y%m%d%H%M%S',errors='coerce')

def timestamp_ns(values):
    # pandas versions can store datetime64[us] or datetime64[ns]. Normalize explicitly.
    return np.asarray(values,dtype='datetime64[ns]').astype('int64')

def load_export(path):
    with zipfile.ZipFile(path) as z:
        out={name:pd.read_csv(z.open('export/'+name+'.csv'),dtype=str,encoding='utf-8') for name in ['charger','status_event','collect_log']}
    return out['charger'],out['status_event'],out['collect_log']

def complete_runs(log):
    records=[]
    for (api,start),g in log.groupby(['api','started_at'],sort=True):
        pages=pd.to_numeric(g.page,errors='coerce'); totals=pd.to_numeric(g.total_count,errors='coerce')
        items=pd.to_numeric(g.n_items,errors='coerce')
        expected=max(1,int(np.ceil(totals.max()/9999))) if totals.notna().all() else -1
        valid=(g.ok.eq('1').all() and totals.nunique()==1 and pages.nunique()==len(g) and sorted(pages.tolist())==list(range(1,expected+1)) and items.sum()==totals.iloc[0])
        records.append({'api':api,'started_at':start,'knownAt':parse_ts(g.finished_at).max(),'runStart':parse_ts(pd.Series([start])).iloc[0],'complete':bool(valid),'pages':len(g),'total':int(totals.max()) if totals.notna().any() else None,'items':int(items.sum())})
    return pd.DataFrame(records)

def prepare_events(c,e,l):
    runs=complete_runs(l)
    c=c.copy();e=e.copy()
    if c.duplicated(['statId','chgerId']).any(): raise ValueError('Duplicate charger key')
    c['chargerKey']=c.statId+'|'+c.chgerId
    c=c.reset_index(drop=True);c['chargerIndex']=np.arange(len(c))
    e=e.merge(c[['statId','chgerId','chargerIndex']],on=['statId','chgerId'],how='left',validate='many_to_one')
    if e.chargerIndex.isna().any(): raise ValueError('Event without charger metadata')
    e=e.merge(runs[['api','started_at','knownAt','complete']],left_on=['source','collected_at'],right_on=['api','started_at'],how='left',validate='many_to_one')
    if e.knownAt.isna().any(): raise ValueError('Event cannot be matched to collection run')
    # Visibility starts after the whole run completed; never backdate using statUpdDt.
    e['knownAt']=pd.concat([e.knownAt,parse_ts(e.collected_at)],axis=1).max(axis=1)
    e['providerAt']=parse_ts(e.statUpdDt);e['chargeStart']=parse_ts(e.nowTsdt)
    e['initial']=e.prev_stat.isna()
    e['statCode']=pd.to_numeric(e.stat,errors='coerce').fillna(9).astype(int)
    e['idNumber']=pd.to_numeric(e.id,errors='raise')
    e=e.sort_values(['knownAt','idNumber']).reset_index(drop=True)
    previous=e.groupby('chargerIndex').stat.shift(1)
    mismatch=(~e.initial)&previous.notna()&e.prev_stat.ne(previous)
    audit={'charger_rows':len(c),'stations_all':int(c.statId.nunique()),'events':len(e),'initial_states':int(e.initial.sum()),'actual_recorded_changes':int((~e.initial).sum()),'duplicate_event_ids':int(e.id.duplicated().sum()),'event_chain_mismatch':int(mismatch.sum()),'future_provider_timestamps':int((e.providerAt>e.knownAt).sum()),'provider_timestamp_missing':int(e.providerAt.isna().sum()),'runs':int(len(runs)),'incomplete_runs':int((~runs.complete).sum()),'log_rows':len(l),'failed_pages':int(l.ok.ne('1').sum()),'collection_start':str(parse_ts(l.started_at).min()),'collection_end':str(parse_ts(l.finished_at).max()),'initial_inventory_rows':int(c.first_seen.eq(c.first_seen.min()).sum()),'new_inventory_rows':int(c.first_seen.ne(c.first_seen.min()).sum())}
    if mismatch.any(): raise ValueError('Event chain ambiguity; investigate before reconstructing states')
    return c,e,runs,audit

def collection_quality(grid,runs):
    good=runs[runs.complete]
    def age(api):
        t=np.sort(good.loc[good.api.eq(api),'knownAt'].values.astype('datetime64[ns]').astype('int64'))
        now=timestamp_ns(grid); pos=np.searchsorted(t,now,side='right')-1
        return np.where(pos>=0,(now-t[np.maximum(pos,0)])/1e9,np.inf)
    status_age=age('status');info_age=age('info')
    valid=(status_age<=450)&(info_age<=4500)
    # An incomplete run makes its interval unknown, until a later complete full snapshot.
    failed_in_progress = np.zeros(len(grid), dtype=bool)
    for r in runs[~runs.complete].itertuples():
        later=good.loc[good.api.eq('info')&good.knownAt.gt(r.knownAt),'knownAt']
        end=later.min() if len(later) else grid.max()+pd.Timedelta(days=1)
        valid&=~((grid>=r.knownAt)&(grid<end))
        # Offline labels also exclude the interval before the failure was known.
        failed_in_progress |= (grid >= r.runStart) & (grid < r.knownAt)
    coverage=pd.Series(valid.astype(float)).rolling(7,min_periods=7).mean().to_numpy()
    return valid & ~failed_in_progress, coverage


NAMES = CORE + HISTORY
NEVER = np.iinfo(np.int64).min
ROOT = Path(__file__).resolve().parent


def replay_profile(c, e, grid, good, coverage, eligible, profile, types, index, out):
    mask = eligible & (np.ones(len(c), dtype=bool) if types is None else c.chgerType.isin(types).to_numpy())
    stations = np.sort(c.loc[mask, "statId"].unique()).astype(str)
    code = c.statId.map(dict(zip(stations, np.arange(len(stations))))).fillna(-1).to_numpy(int)
    n = len(stations)
    state = np.zeros(len(c), dtype=np.int16)
    charge_started = np.full(len(c), NEVER, dtype=np.int64)
    cursor = 0
    event_times = timestamp_ns(e.knownAt)
    event_charger = e.chargerIndex.to_numpy(int)
    event_state = e.statCode.to_numpy(int)
    event_start = timestamp_ns(e.chargeStart)
    event_is_change = ~e.initial.to_numpy()
    clock = timestamp_ns(grid)
    limits = np.searchsorted(event_times, clock, side="right")
    low = np.searchsorted(event_times, clock - 30 * 60 * 10**9, side="right")
    kw = pd.to_numeric(c.output, errors="coerce").to_numpy(float)
    kw_valid = np.isfinite(kw) & (kw > 0)
    kw = np.where(kw_valid, kw, 0)
    ac = c.chgerType.eq('02').to_numpy()
    dc = c.chgerType.isin(PROFILES["DC_COMBO"]).to_numpy()
    avail_history = np.empty((7, n), dtype=np.int32)
    charge_history = np.empty_like(avail_history)
    known_history = np.zeros_like(avail_history, dtype=bool)
    x = np.lib.format.open_memmap(out / f"{profile}_features.npy", mode="w+", dtype=np.float32,
                                  shape=(len(grid), n, len(NAMES)))
    usable = np.lib.format.open_memmap(out / f"{profile}_usable.npy", mode="w+", dtype=bool, shape=(len(grid), n))
    truth = np.lib.format.open_memmap(out / f"{profile}_truth.npy", mode="w+", dtype=np.int8, shape=(len(grid), n))
    np.save(out / f"{profile}_stations.npy", stations)

    for t, now in enumerate(clock):
        stop = limits[t]
        # Assign in event order, including reports from earlier pages of an incomplete run.
        for i in range(cursor, stop):
            k = event_charger[i]
            state[k], charge_started[k] = event_state[i], event_start[i]
        cursor = stop
        active = np.flatnonzero(mask & (state != 0))
        codes = code[active]
        active_state = state[active]

        def sums(weights=None):
            return np.bincount(codes, weights=weights, minlength=n)

        total = sums()
        available = sums(active_state == 2)
        charging = sums(active_state == 3)
        other = sums(np.isin(active_state, [4, 5, 6]))
        unknown = total - available - charging - other
        known = (total > 0) & (unknown == 0)
        truth[t] = np.where(known & good[t], available >= 1, -1)
        ring = t % 7
        avail_history[ring] = available
        charge_history[ring] = charging
        known_history[ring] = known
        usable[t] = (t >= 6) & (coverage[t] == 1) & known_history.all(axis=0)
        if t < 6:
            x[t] = 0
            continue

        starts = charge_started[active]
        starts = np.where(starts == NEVER, now, starts)
        elapsed = (now - starts) / 60e9
        valid_elapsed = (active_state == 3) & (charge_started[active] != NEVER) & (elapsed >= 0) & (elapsed <= 2880)
        elapsed = np.where(valid_elapsed, elapsed, 0)
        elapsed_n = sums(valid_elapsed)
        elapsed_max = np.zeros(n)
        np.maximum.at(elapsed_max, codes, elapsed)
        changed = event_charger[low[t]:stop][event_is_change[low[t]:stop]]
        changes = np.bincount(code[changed[mask[changed]]], minlength=n)
        output_n = sums(kw_valid[active])
        hour = grid[t].hour + grid[t].minute / 60
        dow = grid[t].dayofweek

        def ratio(num, den):
            return np.divide(num, den, out=np.zeros(n), where=den > 0)

        x[t] = np.column_stack([
            total, available, charging, other, unknown,
            np.full(n, np.sin(2 * np.pi * hour / 24)), np.full(n, np.cos(2 * np.pi * hour / 24)),
            np.full(n, np.sin(2 * np.pi * dow / 7)), np.full(n, np.cos(2 * np.pi * dow / 7)),
            np.full(n, index), ratio(available, total), ratio(charging, total),
            avail_history[(t - 1) % 7], avail_history[(t - 3) % 7], avail_history[(t - 6) % 7],
            charge_history[(t - 3) % 7], avail_history.mean(axis=0), avail_history.min(axis=0),
            available - avail_history[(t - 6) % 7], changes, ratio(changes, total),
            ratio(sums(elapsed), elapsed_n), elapsed_max, ratio(elapsed_n, charging),
            ratio(sums(kw[active]), output_n), ratio(sums(ac[active]), total), ratio(sums(dc[active]), total),
        ])
    x.flush()
    usable.flush()
    truth.flush()
    summary = {"stations": n, "valid_input_rows": int(usable.sum())}
    print("REPLAY", profile, summary, flush=True)
    return summary


def build(zip_path, out, cutoff=None):
    out.mkdir(parents=True, exist_ok=True)
    c, e, runs, audit = prepare_events(*load_export(zip_path))
    if audit["duplicate_event_ids"]:
        raise ValueError("Duplicate event identifiers")
    if cutoff is not None:
        e = e.loc[e.knownAt.le(cutoff)].reset_index(drop=True)
        runs = runs.loc[runs.knownAt.le(cutoff)].copy()
    eligible = c.limitYn.eq("N") & c.delYn.eq("N")
    eligible &= pd.to_numeric(c.lat, errors="coerce").between(-90, 90)
    eligible &= pd.to_numeric(c.lng, errors="coerce").between(-180, 180)
    grid = pd.date_range(e.knownAt.min().ceil("5min"), runs.knownAt.max().floor("5min"), freq="5min")
    good, coverage = collection_quality(grid, runs)
    np.save(out / "clock.npy", grid.to_numpy(dtype="datetime64[m]"))
    np.save(out / "collector_good.npy", good)
    audit.update(source_zip_sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                 grid_start=str(grid.min()), grid_end=str(grid.max()), grid_points=len(grid),
                 usable_collector_ticks=int(good.sum()), public_nondeleted_chargers=int(eligible.sum()),
                 profile_stats={})
    for index, (profile, types) in enumerate(PROFILES.items()):
        audit["profile_stats"][profile] = replay_profile(c, e, grid, good, coverage, eligible.to_numpy(),
                                                         profile, types, index, out)
    save_json(out / "data_audit.json", audit)
    runs.to_csv(out / "collection_runs.csv", index=False)
    print("REPLAY_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "state_replay")
    args = parser.parse_args()
    build(args.zip_path, args.out)
