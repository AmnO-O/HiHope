import { AlignedSpans, Span, TokenSpan } from '../types';

const INFLECTION_SUFFIXES = [
  "’s", "'s",
  "ses", "sen", "se",
  "es", "ed", "ing",
  "en", "ern", "er",
  "s", "e", "n"
];

const FUGEN_ELEMENTS = ['ens', 'es', 'en', 'er', 's', 'n', 'e', '-'];

const GERMAN_PARTICLES = [
  "ab", "an", "auf", "aus", "bei", "ein", "fest", "fort", "her", "hin",
  "los", "mit", "nach", "vor", "weg", "zu", "zurück", "zusammen"
];

const IRREGULAR_DE: Record<string, string[]> = {
  "biegen": ["bog", "böge", "gebogen", "biegt", "biegst"],
  "bieten": ["bot", "böte", "geboten", "bietet", "bietest"],
  "bleiben": ["blieb", "geblieben", "bleibt", "bleibst", "bliebe"],
  "brennen": ["brannt", "brannte", "brannten", "gebrannt", "brennt", "brennst"],
  "bringen": ["bracht", "brachte", "brachten", "gebracht", "bringt", "bringst"],
  "denken": ["dacht", "dachte", "dachten", "gedacht", "denkt", "denkst"],
  "empfehlen": ["empfahl", "empfohlen", "empfiehlt", "empfiehlst"],
  "essen": ["aß", "gegessen", "isst", "iss"],
  "fahren": ["fuhr", "gefahren", "fährt", "fährst"],
  "fallen": ["fiel", "gefallen", "fällt", "fällst"],
  "fangen": ["fing", "gefangen", "fängt", "fängst"],
  "finden": ["fand", "gefunden", "findet", "findest"],
  "fliegen": ["flog", "geflogen", "fliegt", "fliegst"],
  "fliehen": ["floh", "geflohen", "flieht", "fliehst"],
  "fließen": ["floss", "geflossen", "fließt"],
  "geben": ["gab", "gegeben", "gibt", "gibst", "gib"],
  "gehen": ["ging", "gegangen", "geht", "gehst"],
  "haben": ["hatte", "gehabt", "hat", "hast"],
  "halten": ["hielt", "gehalten", "hält", "hältst"],
  "hängen": ["hing", "gehangen", "hängt"],
  "hauen": ["hieb", "gehauen", "haut", "haust"],
  "heißen": ["hieß", "geheißen", "heißt"],
  "helfen": ["half", "geholfen", "hilft", "hilfst"],
  "kennen": ["kannte", "gekannt", "kennt"],
  "kommen": ["kam", "gekommen", "kommt", "kommst"],
  "kriechen": ["kroch", "gekrochen", "kriecht"],
  "laden": ["lud", "geladen", "lädt", "lädst"],
  "lassen": ["ließ", "gelassen", "lässt", "lasst"],
  "laufen": ["lief", "gelaufen", "läuft", "läufst"],
  "lesen": ["las", "gelesen", "liest"],
  "liegen": ["lag", "gelegen", "liegt", "liegst"],
  "nehmen": ["nahm", "genommen", "nimmt", "nimmst", "nimm"],
  "nennen": ["nannte", "genannt", "nennt"],
  "raten": ["riet", "geraten", "rät", "rätst"],
  "reißen": ["riss", "gerissen", "reißt"],
  "reiten": ["ritt", "geritten", "reitet"],
  "rennen": ["rannte", "gerannt", "rennt"],
  "rufen": ["rief", "gerufen", "ruft"],
  "schaffen": ["schuf", "geschaffen", "schafft"],
  "scheinen": ["schien", "geschienen", "scheint"],
  "schieben": ["schob", "geschoben", "schiebt"],
  "schießen": ["schoss", "geschossen", "schießt"],
  "schlafen": ["schlief", "geschlafen", "schläft"],
  "schlagen": ["schlug", "geschlagen", "schlägt"],
  "schließen": ["schloss", "geschlossen", "schließt"],
  "schneiden": ["schnitt", "geschnitten", "schneidet"],
  "schreiben": ["schrieb", "geschrieben", "schreibt"],
  "schreien": ["schrie", "geschrien", "schreit"],
  "sehen": ["sah", "gesehen", "sieht", "siehst", "sieh"],
  "sein": ["war", "gewesen", "ist", "bist", "sind"],
  "senden": ["sandte", "gesandt", "sendet"],
  "singen": ["sang", "gesungen", "singt"],
  "sinken": ["sank", "gesunken", "sinkt"],
  "sitzen": ["saß", "gesessen", "sitzt"],
  "sprechen": ["sprach", "gesprochen", "spricht", "sprichst"],
  "springen": ["sprang", "gesprungen", "springt"],
  "stehen": ["stand", "gestanden", "steht", "stehst"],
  "stehlen": ["stahl", "gestohlen", "stiehlt"],
  "steigen": ["stieg", "gestiegen", "steigt"],
  "sterben": ["starb", "gestorben", "stirbt"],
  "stinken": ["stank", "gestunken", "stinkt"],
  "tragen": ["trug", "getragen", "trägt", "trägst"],
  "treffen": ["traf", "getroffen", "trifft", "triffst"],
  "treiben": ["trieb", "getrieben", "treibt"],
  "treten": ["trat", "getreten", "tritt", "trittst"],
  "trinken": ["trank", "getrunken", "trinkt"],
  "tun": ["tat", "getan", "tut"],
  "vergessen": ["vergaß", "vergessen", "vergisst"],
  "verlieren": ["verlor", "verloren", "verliert"],
  "wachsen": ["wuchs", "gewachsen", "wächst"],
  "waschen": ["wusch", "gewaschen", "wäscht"],
  "weichen": ["wich", "gewichen", "weicht"],
  "weisen": ["wies", "gewiesen", "weist"],
  "wenden": ["wandte", "gewandt", "wendet"],
  "werben": ["warb", "geworben", "wirbt"],
  "werden": ["wurde", "geworden", "wird", "wirst"],
  "werfen": ["warf", "geworfen", "wirft", "wirfst"],
  "wiegen": ["wog", "gewogen", "wiegt"],
  "wissen": ["wusste", "gewusst", "weiß", "weißt"],
  "ziehen": ["zog", "gezogen", "zieht"],
  "zwingen": ["zwang", "gezwungen", "zwingt"]
};

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Marker-Free Compound and Span Matcher (MoTune src/marks.py algorithm)
 */
export function alignSpansInContext(
  context: string,
  modBase: string,
  headBase: string
): AlignedSpans {
  const ctx = context;
  const modClean = modBase.trim();
  const headClean = headBase.trim();

  if (!modClean || !headClean || !ctx) {
    return {
      modSpan: { start: null, end: null },
      headSpan: { start: null, end: null },
      compoundSpan: { start: null, end: null },
      degenerate: false,
      matchType: 'none',
      fugenDetected: null,
      separableVerbFound: false,
    };
  }

  // 1. STRATEGY 1: Fused German Compound ("Abiturzeugnis", "Staubsauger")
  for (const fugen of FUGEN_ELEMENTS) {
    for (const infl of ['', ...INFLECTION_SUFFIXES]) {
      const pattern = new RegExp(
        `(^|[^\\p{L}\\d])(${escapeRegex(modClean)})(${fugen === '-' ? '-' : escapeRegex(fugen)})(${escapeRegex(headClean)}${escapeRegex(infl)})(?=[^\\p{L}\\d]|$)`,
        'gui'
      );
      const match = pattern.exec(ctx);
      if (match) {
        const fullMatchedStr = match[0];
        const prefixLen = match[1].length;
        const matchStart = match.index + prefixLen;

        const modStr = match[2];
        const fugenStr = match[3];
        const headStr = match[4];

        const modStart = matchStart;
        const modEnd = modStart + modStr.length;
        const headStart = modEnd + fugenStr.length;
        const headEnd = headStart + headStr.length;

        return {
          modSpan: { start: modStart, end: modEnd },
          headSpan: { start: headStart, end: headEnd },
          compoundSpan: { start: modStart, end: headEnd },
          degenerate: false,
          matchType: 'fused',
          fugenDetected: fugenStr.length > 0 ? fugenStr : null,
          separableVerbFound: false,
        };
      }
    }
  }

  // 2. STRATEGY 2: Spaced Compound ("night watch", "flea market", "paper bags")
  for (const infl of ['', ...INFLECTION_SUFFIXES]) {
    const pattern = new RegExp(
      `(^|[^\\p{L}\\d])(${escapeRegex(modClean)})\\s+(${escapeRegex(headClean)}${escapeRegex(infl)})(?=[^\\p{L}\\d]|$)`,
      'gui'
    );
    const match = pattern.exec(ctx);
    if (match) {
      const prefixLen = match[1].length;
      const matchStart = match.index + prefixLen;
      const modStr = match[2];
      const headStr = match[3];

      const modStart = matchStart;
      const modEnd = modStart + modStr.length;
      const headIndex = ctx.indexOf(headStr, modEnd);
      const headEnd = headIndex + headStr.length;

      return {
        modSpan: { start: modStart, end: modEnd },
        headSpan: { start: headIndex, end: headEnd },
        compoundSpan: { start: modStart, end: headEnd },
        degenerate: false,
        matchType: 'spaced',
        fugenDetected: null,
        separableVerbFound: false,
      };
    }
  }

  // 3. STRATEGY 3: German Separable Verbs (Discontinuous, e.g. "haut ... ab")
  const lowerMod = modClean.toLowerCase();
  const lowerHead = headClean.toLowerCase();
  let verbCandidates = [lowerHead, ...INFLECTION_SUFFIXES.map(s => lowerHead + s)];
  if (IRREGULAR_DE[lowerHead]) {
    verbCandidates.push(...IRREGULAR_DE[lowerHead]);
  }
  // If mod is particle (e.g. "ab") and head is verb (e.g. "hauen") or vice-versa
  const isModParticle = GERMAN_PARTICLES.includes(lowerMod);

  if (isModParticle) {
    for (const vCand of verbCandidates) {
      const vRegex = new RegExp(`\\b${escapeRegex(vCand)}\\b`, 'gui');
      const pRegex = new RegExp(`\\b${escapeRegex(modClean)}\\b`, 'gui');
      const vMatch = vRegex.exec(ctx);
      const pMatch = pRegex.exec(ctx);

      if (vMatch && pMatch) {
        const vStart = vMatch.index;
        const vEnd = vStart + vMatch[0].length;
        const pStart = pMatch.index;
        const pEnd = pStart + pMatch[0].length;
        const compStart = Math.min(vStart, pStart);
        const compEnd = Math.max(vEnd, pEnd);

        return {
          modSpan: { start: pStart, end: pEnd },
          headSpan: { start: vStart, end: vEnd },
          compoundSpan: { start: compStart, end: compEnd },
          degenerate: false,
          matchType: 'independent',
          fugenDetected: null,
          separableVerbFound: true,
        };
      }
    }
  }

  // 4. STRATEGY 4: Independent Search (Modifier then Head anywhere)
  const modIdx = ctx.toLowerCase().indexOf(modClean.toLowerCase());
  if (modIdx !== -1) {
    const headIdx = ctx.toLowerCase().indexOf(headClean.toLowerCase(), modIdx + modClean.length);
    if (headIdx !== -1) {
      return {
        modSpan: { start: modIdx, end: modIdx + modClean.length },
        headSpan: { start: headIdx, end: headIdx + headClean.length },
        compoundSpan: { start: modIdx, end: headIdx + headClean.length },
        degenerate: false,
        matchType: 'independent',
        fugenDetected: null,
        separableVerbFound: false,
      };
    }
  }

  return {
    modSpan: { start: null, end: null },
    headSpan: { start: null, end: null },
    compoundSpan: { start: null, end: null },
    degenerate: false,
    matchType: 'none',
    fugenDetected: null,
    separableVerbFound: false,
  };
}

/**
 * Breaks down context into rendered TokenSpan chunks with exact roles
 */
export function buildTokenChunks(context: string, spans: AlignedSpans): TokenSpan[] {
  if (!context) return [];
  const { modSpan, headSpan, fugenDetected } = spans;

  const boundaries: { index: number; type: 'start' | 'end'; role: TokenSpan['role'] }[] = [];

  if (modSpan.start !== null && modSpan.end !== null) {
    boundaries.push({ index: modSpan.start, type: 'start', role: 'mod' });
    boundaries.push({ index: modSpan.end, type: 'end', role: 'mod' });
  }

  if (fugenDetected && modSpan.end !== null && headSpan.start !== null && modSpan.end < headSpan.start) {
    boundaries.push({ index: modSpan.end, type: 'start', role: 'fugen' });
    boundaries.push({ index: headSpan.start, type: 'end', role: 'fugen' });
  }

  if (headSpan.start !== null && headSpan.end !== null) {
    boundaries.push({ index: headSpan.start, type: 'start', role: 'head' });
    boundaries.push({ index: headSpan.end, type: 'end', role: 'head' });
  }

  boundaries.sort((a, b) => a.index - b.index);

  const chunks: TokenSpan[] = [];
  let currIdx = 0;
  let activeRole: TokenSpan['role'] = 'none';

  for (const b of boundaries) {
    if (b.index > currIdx) {
      chunks.push({
        text: context.slice(currIdx, b.index),
        start: currIdx,
        end: b.index,
        role: activeRole,
      });
      currIdx = b.index;
    }
    activeRole = b.type === 'start' ? b.role : 'none';
  }

  if (currIdx < context.length) {
    chunks.push({
      text: context.slice(currIdx),
      start: currIdx,
      end: context.length,
      role: 'none',
    });
  }

  return chunks;
}
