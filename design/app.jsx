// Compact app — 480 × 720 — Jantama AI Assistant
// Three recommendation_style layouts: 'hero' | 'compare' | 'mini'
// All states: dahai / riichi / fuuro / agari
// LLM reason (S-01) and danger/safe (S-02) included where relevant.

const { useState, useMemo } = React;

// ── Top status bar ────────────────────────────────────────────────────────
function StatusBar({ monitoring = true }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '10px 14px', borderBottom: '1px solid var(--border-1)',
      background: 'rgba(255,255,255,0.92)', backdropFilter: 'blur(12px) saturate(160%)',
      WebkitBackdropFilter: 'blur(12px) saturate(160%)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          fontFamily: 'var(--font-sans)', fontWeight: 900, fontSize: 14,
          letterSpacing: '-0.02em', display: 'flex', alignItems: 'baseline', gap: 1,
        }}>
          <span className="gradient-text" style={{ fontSize: 17 }}>雀</span>
          <span style={{ color: 'var(--ink-900)' }}>tama</span>
          <span style={{ color: 'var(--ink-400)', fontWeight: 500, marginLeft: 6, fontSize: 11, letterSpacing: '0.12em' }}>AI</span>
        </div>
        <div style={{ width: 1, height: 12, background: 'var(--border-1)' }} />
        <MonitorPill on={monitoring} />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--ink-500)', fontFamily: 'var(--font-mono)' }}>
        <span>14:32:08</span>
        <span style={{ width: 1, height: 10, background: 'var(--border-1)' }} />
        <button title="設定" style={iconBtnStyle}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        </button>
      </div>
    </div>
  );
}

const iconBtnStyle = {
  width: 22, height: 22, borderRadius: 4, border: 'none', background: 'transparent',
  color: 'var(--ink-600)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
};

function MonitorPill({ on }) {
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '3px 8px 3px 6px', borderRadius: 999,
      background: on ? '#0F0F1E' : 'var(--ink-100)',
      color: on ? '#FAFAF7' : 'var(--ink-600)',
      fontSize: 10, fontFamily: 'var(--font-sans)', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase',
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: 999,
        background: on ? '#FF2600' : 'var(--ink-400)',
        boxShadow: on ? '0 0 0 3px rgba(255,38,0,0.2)' : 'none',
        animation: on ? 'jt-pulse 1.4s ease-in-out infinite' : 'none',
      }} />
      <span>{on ? 'LIVE' : 'OFF'}</span>
    </div>
  );
}

// ── Game context bar (round / turn / dora) ───────────────────────────────
function ContextBar() {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '8px 14px', background: 'var(--ink-50)',
      borderBottom: '1px solid var(--border-2)', fontFamily: 'var(--font-sans)', fontSize: 11,
    }}>
      <div style={{ display: 'flex', gap: 14, color: 'var(--ink-600)' }}>
        <span><strong style={{ color: 'var(--ink-900)', fontWeight: 700 }}>東1局</strong> 6巡目</span>
        <span>自風 <strong style={{ color: 'var(--ink-900)', fontWeight: 700 }}>東</strong></span>
        <span>持点 <strong style={{ color: 'var(--ink-900)', fontWeight: 700, fontFeatureSettings: '"tnum"' }}>25,000</strong></span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--ink-600)' }}>
        <span style={{ fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Dora</span>
        <Tile code="5p" size="xs" />
      </div>
    </div>
  );
}

function ConfBar({ value, color = 'var(--ink-900)', height = 4 }) {
  return (
    <div style={{ width: '100%', height, background: 'var(--ink-100)', borderRadius: 999, overflow: 'hidden' }}>
      <div style={{ width: `${Math.max(0, Math.min(100, value * 100))}%`, height: '100%', background: color, borderRadius: 999, transition: 'width 0.3s var(--ease-out)' }} />
    </div>
  );
}

function ActionPill({ kind }) {
  const map = {
    'リーチ':  { bg: 'linear-gradient(135deg, #0432FF, #FF2600)', fg: '#fff' },
    '和了':    { bg: '#138A4F', fg: '#fff' },
    'ロン':    { bg: '#138A4F', fg: '#fff' },
    'ツモ':    { bg: '#138A4F', fg: '#fff' },
    '打牌':    { bg: 'var(--ink-900)', fg: '#fff' },
    'ポン':    { bg: '#1F1F2E', fg: '#fff' },
    'チー':    { bg: '#1F1F2E', fg: '#fff' },
    'カン':    { bg: '#1F1F2E', fg: '#fff' },
    'スルー':  { bg: 'var(--ink-100)', fg: 'var(--ink-900)' },
    'ダマ':    { bg: 'var(--ink-100)', fg: 'var(--ink-900)' },
    '見逃し':  { bg: 'var(--ink-100)', fg: 'var(--ink-500)' },
    '鳴き選択': { bg: 'var(--ink-900)', fg: '#fff' },
  };
  const s = map[kind] || map['打牌'];
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', padding: '3px 10px',
      borderRadius: 999, background: s.bg, color: s.fg,
      fontFamily: 'var(--font-sans)', fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase',
    }}>{kind}</span>
  );
}

function PrimaryGlyph({ scenario, size = 'xl' }) {
  const t = scenario.primary.tile;
  if (/^[1-9][mpsz]$/.test(t)) return <Tile code={t} size={size} highlight />;
  return <VerbCard label={t} size={size} />;
}

function VerbCard({ label, size = 'xl' }) {
  const dims = { md: 56, lg: 80, xl: 92, xxl: 140 }[size] || 92;
  const isGradient = ['リーチ', 'ロン', 'ツモ'].includes(label);
  return (
    <div style={{
      width: dims, height: dims * 1.3, borderRadius: 10,
      background: isGradient ? 'linear-gradient(135deg, #0432FF 0%, #FF2600 100%)' : '#0F0F1E',
      color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'var(--font-jp)', fontWeight: 800, fontSize: dims * 0.32,
      letterSpacing: '0.02em',
      boxShadow: '0 12px 40px rgba(15,15,30,0.18), 0 2px 6px rgba(15,15,30,0.08)',
    }}>{label}</div>
  );
}

function HeroLayout({ scenario }) {
  const top = scenario.candidates[0];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 14,
        padding: '14px 16px',
        background: 'var(--white)', borderRadius: 12,
        border: '1px solid var(--border-1)', boxShadow: 'var(--shadow-2)',
        position: 'relative', overflow: 'hidden',
      }}>
        <div style={{ position: 'absolute', inset: 0, background: 'var(--gradient-acial-halo)', opacity: 0.5, pointerEvents: 'none' }} />
        <div style={{ position: 'relative', zIndex: 1 }}>
          <PrimaryGlyph scenario={scenario} size="xl" />
        </div>
        <div style={{ position: 'relative', zIndex: 1, flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
            <ActionPill kind={scenario.actionType} />
            <span style={{ fontSize: 9, color: 'var(--ink-500)', fontFamily: 'var(--font-sans)', letterSpacing: '0.08em', fontWeight: 700 }}>RANK 1</span>
          </div>
          <div style={{ font: 'var(--text-h4)', color: 'var(--ink-900)', lineHeight: 1.2, marginBottom: 6 }}>
            {scenario.primary.label}
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
            <span style={{ fontSize: 9, color: 'var(--ink-500)', fontFamily: 'var(--font-sans)', letterSpacing: '0.1em', fontWeight: 700 }}>EV</span>
            <span className="gradient-text" style={{ fontSize: 26, fontWeight: 800, fontFamily: 'var(--font-sans)', letterSpacing: '-0.02em', fontFeatureSettings: '"tnum"' }}>{scenario.primary.score}</span>
          </div>
          <div style={{ marginTop: 6 }}>
            <ConfBar value={top.prob} color="var(--ink-900)" />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3, fontSize: 9, color: 'var(--ink-500)', fontFamily: 'var(--font-sans)', letterSpacing: '0.04em' }}>
              <span>確信度</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{Math.round(top.prob * 100)}%</span>
            </div>
          </div>
        </div>
      </div>
      <CandidateList candidates={scenario.candidates.slice(1)} startRank={2} />
    </div>
  );
}

function CandidateList({ candidates, startRank = 2 }) {
  if (!candidates.length) return null;
  return (
    <div>
      <div style={{ font: 'var(--text-overline)', color: 'var(--ink-500)', marginBottom: 8, paddingLeft: 2 }}>次点候補</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {candidates.map((c, i) => (
          <div key={i} style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '8px 12px', background: 'var(--white)',
            borderRadius: 8, border: '1px solid var(--border-1)',
          }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-400)', width: 14 }}>0{startRank + i}</span>
            <div style={{ width: 30, display: 'flex', justifyContent: 'center' }}>
              {/^[1-9][mpsz]$/.test(c.tile)
                ? <Tile code={c.tile} size="sm" />
                : <span style={{ fontFamily: 'var(--font-jp)', fontWeight: 700, fontSize: 13, color: 'var(--ink-700)' }}>{c.action || c.tile}</span>}
            </div>
            <div style={{ flex: 1 }}>
              <ConfBar value={c.prob} color="var(--ink-300)" height={3} />
            </div>
            <div style={{ fontFamily: 'var(--font-sans)', fontWeight: 700, fontSize: 13, color: c.ev >= 0 ? 'var(--ink-900)' : 'var(--ink-500)', fontFeatureSettings: '"tnum"', minWidth: 50, textAlign: 'right' }}>
              {c.ev >= 0 ? '+' : ''}{c.ev.toFixed(2)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CompareLayout({ scenario }) {
  const cands = scenario.candidates.slice(0, 3);
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <ActionPill kind={scenario.actionType} />
        <span style={{ font: 'var(--text-overline)', color: 'var(--ink-500)' }}>TOP {cands.length}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cands.length}, 1fr)`, gap: 8 }}>
        {cands.map((c, i) => {
          const isTop = i === 0;
          const isTile = /^[1-9][mpsz]$/.test(c.tile);
          return (
            <div key={i} style={{
              padding: '14px 8px 12px',
              borderRadius: 10,
              background: isTop ? '#0F0F1E' : 'var(--white)',
              color: isTop ? '#FAFAF7' : 'var(--ink-900)',
              border: isTop ? 'none' : '1px solid var(--border-1)',
              boxShadow: isTop ? '0 12px 30px rgba(15,15,30,0.20)' : 'var(--shadow-1)',
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
              position: 'relative', overflow: 'hidden',
            }}>
              {isTop && <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: 'linear-gradient(135deg, #0432FF, #FF2600)' }} />}
              <span style={{ fontFamily: 'var(--font-sans)', fontSize: 9, fontWeight: 700, letterSpacing: '0.12em', color: isTop ? 'var(--ink-300)' : 'var(--ink-400)' }}>
                {i === 0 ? 'BEST' : `#${i + 1}`}
              </span>
              <div style={{ height: 64, display: 'flex', alignItems: 'center' }}>
                {isTile
                  ? <Tile code={c.tile} size="lg" highlight={isTop} />
                  : <div style={{
                      padding: '8px 14px', borderRadius: 6, background: isTop ? 'rgba(255,255,255,0.10)' : 'var(--ink-50)',
                      fontFamily: 'var(--font-jp)', fontSize: 18, fontWeight: 800, color: isTop ? '#FAFAF7' : 'var(--ink-900)',
                    }}>{c.action || c.tile}</div>}
              </div>
              <div style={{
                fontFamily: 'var(--font-sans)', fontSize: 18, fontWeight: 800,
                fontFeatureSettings: '"tnum"', letterSpacing: '-0.02em',
                color: isTop ? '#FAFAF7' : (c.ev >= 0 ? 'var(--ink-900)' : 'var(--ink-500)'),
              }}>{c.ev >= 0 ? '+' : ''}{c.ev.toFixed(2)}</div>
              <div style={{ width: '80%' }}>
                <ConfBar value={c.prob} color={isTop ? '#FAFAF7' : 'var(--ink-300)'} height={3} />
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: isTop ? 'var(--ink-300)' : 'var(--ink-500)' }}>
                {Math.round(c.prob * 100)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MiniLayout({ scenario }) {
  const cands = scenario.candidates.slice(0, 4);
  return (
    <div>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '10px 12px', background: '#0F0F1E', color: '#FAFAF7',
        borderRadius: 8, marginBottom: 8, position: 'relative', overflow: 'hidden',
      }}>
        <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: 'linear-gradient(180deg, #0432FF, #FF2600)' }} />
        <ActionPill kind={scenario.actionType} />
        <div style={{ flex: 1, font: 'var(--text-h4)', color: '#FAFAF7' }}>
          {scenario.primary.label}
        </div>
        <div style={{ fontFamily: 'var(--font-sans)', fontWeight: 800, fontSize: 22, letterSpacing: '-0.02em', fontFeatureSettings: '"tnum"' }}>
          {scenario.primary.score}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {cands.map((c, i) => {
          const isTile = /^[1-9][mpsz]$/.test(c.tile);
          return (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '7px 10px',
              borderBottom: i < cands.length - 1 ? '1px dashed var(--border-1)' : 'none',
            }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-400)', width: 14 }}>0{i + 1}</span>
              <div style={{ width: 26, display: 'flex', justifyContent: 'center' }}>
                {isTile
                  ? <Tile code={c.tile} size="xs" highlight={i === 0} />
                  : <span style={{ fontFamily: 'var(--font-jp)', fontWeight: 700, fontSize: 12, color: 'var(--ink-900)' }}>{c.action || c.tile}</span>}
              </div>
              <div style={{ flex: 1 }}>
                <ConfBar value={c.prob} color={i === 0 ? 'var(--ink-900)' : 'var(--ink-300)'} height={2} />
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-500)', width: 32, textAlign: 'right' }}>
                {Math.round(c.prob * 100)}%
              </div>
              <div style={{ fontFamily: 'var(--font-sans)', fontWeight: 700, fontSize: 12, color: c.ev >= 0 ? 'var(--ink-900)' : 'var(--ink-500)', fontFeatureSettings: '"tnum"', minWidth: 44, textAlign: 'right' }}>
                {c.ev >= 0 ? '+' : ''}{c.ev.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ReasonBlock({ scenario }) {
  return (
    <div style={{
      padding: '10px 12px', borderRadius: 8,
      background: 'var(--ink-50)', border: '1px solid var(--border-2)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <span style={{
          width: 16, height: 16, borderRadius: 3,
          background: 'var(--gradient-acial)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          color: '#fff', fontSize: 8, fontWeight: 800, fontFamily: 'var(--font-sans)', letterSpacing: '0.04em',
        }}>AI</span>
        <span style={{ font: 'var(--text-overline)', color: 'var(--ink-500)' }}>Reasoning</span>
      </div>
      <div style={{ font: 'var(--text-body-sm)', color: 'var(--ink-700)', lineHeight: 1.6, textWrap: 'pretty' }}>
        {scenario.reason}
      </div>
    </div>
  );
}

function DangerSafeBlock({ scenario }) {
  if (!scenario.danger || (!scenario.danger.length && !scenario.safe.length)) return null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
      {scenario.danger.length > 0 && (
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--danger-bg)', border: '1px solid rgba(199,27,0,0.18)' }}>
          <div style={{ font: 'var(--text-overline)', color: 'var(--danger)', marginBottom: 6 }}>危険牌</div>
          <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            {scenario.danger.map((d, i) => <Tile key={i} code={d.tile} size="sm" />)}
          </div>
        </div>
      )}
      {scenario.safe.length > 0 && (
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--success-bg)', border: '1px solid rgba(19,138,79,0.18)' }}>
          <div style={{ font: 'var(--text-overline)', color: 'var(--success)', marginBottom: 6 }}>安全牌</div>
          <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            {scenario.safe.map((t, i) => <Tile key={i} code={t} size="sm" />)}
          </div>
        </div>
      )}
    </div>
  );
}

function HandRow({ scenario }) {
  const tiles = handFor(scenario);
  return (
    <div style={{
      padding: '10px 12px', borderTop: '1px solid var(--border-1)',
      background: 'var(--white)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ font: 'var(--text-overline)', color: 'var(--ink-500)' }}>手牌</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-400)' }}>{tiles.length} 牌</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2 }}>
        {tiles.slice(0, 13).map((t, i) => {
          const isReco = scenario.actionType === '打牌' && t === scenario.primary.tile;
          return <Tile key={i} code={t} size="sm" highlight={isReco && i === tiles.indexOf(scenario.primary.tile)} />;
        })}
        {tiles.length === 14 && (
          <>
            <div style={{ width: 4 }} />
            <Tile code={tiles[13]} size="sm" />
          </>
        )}
      </div>
    </div>
  );
}

function MonitorButton({ on = true }) {
  return (
    <button style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
      width: '100%', padding: '12px 16px',
      borderRadius: 8, border: 'none', cursor: 'pointer',
      background: on ? '#0F0F1E' : 'var(--gradient-acial)',
      color: '#FAFAF7',
      fontFamily: 'var(--font-jp)', fontWeight: 700, fontSize: 14, letterSpacing: '0.04em',
    }}>
      <span style={{
        width: 8, height: 8, borderRadius: 999,
        background: on ? '#FF2600' : '#fff',
      }} />
      {on ? '監視中 — 停止する' : '監視を開始する'}
    </button>
  );
}

// ── App shell ─────────────────────────────────────────────────────────────
function JantamaApp({ state = 'dahai', layout = 'hero', showReason = true, showDanger = true }) {
  const scenario = SCENARIOS[state];
  return (
    <div style={{
      width: 480, height: 720, background: 'var(--ink-50)',
      display: 'flex', flexDirection: 'column',
      borderRadius: 12, overflow: 'hidden',
      boxShadow: '0 24px 60px rgba(15,15,30,0.14), 0 4px 12px rgba(15,15,30,0.06)',
      border: '1px solid var(--border-1)',
      fontFamily: 'var(--font-jp)',
    }}>
      <StatusBar monitoring={true} />
      <ContextBar />
      <div style={{ padding: 14, flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {layout === 'hero' && <HeroLayout scenario={scenario} />}
        {layout === 'compare' && <CompareLayout scenario={scenario} />}
        {layout === 'mini' && <MiniLayout scenario={scenario} />}
        {showReason && <ReasonBlock scenario={scenario} />}
        {showDanger && <DangerSafeBlock scenario={scenario} />}
      </div>
      <HandRow scenario={scenario} />
      <div style={{ padding: 12, background: 'var(--white)', borderTop: '1px solid var(--border-1)' }}>
        <MonitorButton on={true} />
      </div>
    </div>
  );
}

window.JantamaApp = JantamaApp;
window.MonitorPill = MonitorPill;
window.ActionPill = ActionPill;
window.Tile = window.Tile;
