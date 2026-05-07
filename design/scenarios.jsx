// Shared mock game state — the "monitoring + dahai recommendation" baseline scenario.

const SCENARIOS = {
  dahai: {
    label: '打牌',
    primary: { tile: '6m', score: '+0.32', delta: 'EV', label: '6m を切る' },
    candidates: [
      { tile: '6m', ev: 0.32, prob: 0.61 },
      { tile: '9p', ev: 0.18, prob: 0.22 },
      { tile: '1z', ev: -0.05, prob: 0.11 },
      { tile: '3s', ev: -0.21, prob: 0.06 },
    ],
    actionType: '打牌',
    handCode: 'tenpai',
    reason: '受け入れ最大の打。6mを切るとピンフ・三色の両天秤で、シャンテン戻しなく17種56牌の有効牌が残る。9pは安全度が一段下がり、ドラ表示牌5pの周辺で他家リーチへの放銃率が上がる。',
    danger: [
      { tile: '7p', level: 'high' },
      { tile: '3z', level: 'mid' },
    ],
    safe: ['1z', '4z', '9s'],
  },
  riichi: {
    label: 'リーチ',
    primary: { tile: 'リーチ', score: '+0.84', delta: 'EV', label: 'リーチを宣言' },
    candidates: [
      { tile: 'riichi', ev: 0.84, prob: 0.78, action: 'リーチ' },
      { tile: '5p', ev: 0.21, prob: 0.14, action: 'ダマ' },
      { tile: '2s', ev: 0.05, prob: 0.08, action: 'ダマ' },
    ],
    actionType: 'リーチ',
    handCode: 'tenpai',
    reason: '良形テンパイ・打点上昇余地・順目に余裕があり、リーチ宣言の期待値が圧倒的に高い。三人の捨牌は無筋少なく、放銃リスクは中庸。裏ドラ・一発の打点込みで+0.84の優位。',
    danger: [],
    safe: [],
  },
  fuuro: {
    label: '鳴き判断',
    primary: { tile: 'スルー', score: '+0.12', delta: 'EV', label: 'ポンせずスルー' },
    candidates: [
      { tile: 'pass', ev: 0.12, prob: 0.58, action: 'スルー' },
      { tile: 'pon', ev: 0.04, prob: 0.27, action: 'ポン' },
      { tile: 'chi', ev: -0.08, prob: 0.15, action: 'チー' },
    ],
    actionType: '鳴き選択',
    handCode: 'menzen',
    reason: '門前を維持した方が、リーチ込みの打点期待値が高い。中をポンしても役は確定するが、形が崩れて受け入れが14牌→9牌に減少。3巡目で局速はまだ十分。',
    danger: [],
    safe: [],
  },
  agari: {
    label: '和了判断',
    primary: { tile: 'ロン', score: '+1.00', delta: 'EV', label: 'ロンで和了' },
    candidates: [
      { tile: 'ron', ev: 1.0, prob: 0.99, action: 'ロン' },
      { tile: 'pass', ev: -0.21, prob: 0.01, action: '見逃し' },
    ],
    actionType: '和了',
    handCode: 'agari',
    reason: 'リーチ・ピンフ・ツモ無し・ドラ1で5,200点。トップ目との点差を考えると見逃しの価値はなく、即和了が最善。残り2局・親番なし。',
    danger: [],
    safe: [],
  },
};

// Hand for the standard scenarios — same physical hand, recommendation differs
const HAND_TENPAI = ['1m','2m','3m','4m','5m','6m','7p','8p','9p','1z','2z','3z','5m'];
const HAND_MENZEN = ['2m','3m','4m','6m','7m','8m','3p','4p','5p','7s','8s','9s','7z'];
const HAND_AGARI  = ['1m','2m','3m','4m','5m','6m','7p','8p','9p','1z','1z','1z','5p']; // 5p ron tile

function handFor(scenario) {
  if (scenario.handCode === 'menzen') return HAND_MENZEN;
  if (scenario.handCode === 'agari') return HAND_AGARI;
  return HAND_TENPAI;
}

const POND_SELF = ['9m','1z','4z','2s'];
const POND_E    = ['7z','9p','1m','5s','2z'];
const POND_S    = ['9s','3p','1z','6s'];
const POND_W    = ['8m','5z','7s','3z','4z','2p'];

const RIVERS = {
  self: POND_SELF,
  east: POND_E,
  south: POND_S,
  west: POND_W,
};

window.SCENARIOS = SCENARIOS;
window.handFor = handFor;
window.RIVERS = RIVERS;
