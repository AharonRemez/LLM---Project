import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert d_out % num_heads == 0, "num_headsחייב להיות מתחלק בd_out "

    self.d_out = d_out # The output embedding size, d_out=2 כרגע
    self.num_heads = num_heads # CausalAttentionמספר ה
    self.head_dim = d_out // num_heads # כמה מימדים כל head יקבל
    # אתחול המשקלים
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    # שכבת איחוד פלט ראשי תשומת לב
    self.out_proj = nn.Linear(d_out, d_out) # השכבה מאחדת את התוצאות מכל ראשי תשומת הלב לפלט יחיד על ידי שילוב ועיבוד התוצאות שלהם לממד אחיד.
    # dropoutשכבת ה
    self.dropout = nn.Dropout(dropout)
    # שכבת הסתרת המידע
    self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

  def forward(self, x):
    b, num_tokens, d_in = x.shape  # משמש כדי לפרק את הצורה (shape) של המטריצה x לשלושה משתנים
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)
    # סידור הנתונים שלנו שיתאפשר לעשות חישובים במקבלים על ראשים
    # לפני היה 3 מימדים(b, num_tokens, d_in) ועכשיו יש 4 (b, num_tokens, self.num_heads, self.head_dim)
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

    keys = keys.transpose(1, 2) # Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
    queries = queries.transpose(1, 2) # Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
    values = values.transpose(1, 2) # Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
    # חישוב ציון הקשב
    attn_scores = queries @ keys.transpose(2, 3) # חישוב ציון הקשב עבור כל ראש
    # ביצוע ההסתרה
    attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
    # חישוב משקל תשומת הלב
    d_k = keys.shape[-1]
    attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1) # החילוק בשורש גורם לנו לשמור על טווח הערכים ריאלי
    # ביצוע dropout למשקלים
    attn_weights = self.dropout(attn_weights)
    # חישוב וקטור ההקשר
    context_vec = (attn_weights @ values).transpose(1, 2) # Tensor shape: (b, num_tokens, n_heads, head_dim) כדי שהחישוב יהיה עבור כל ראש במנגנון
    # contiguous() מסדרת מחדש את הזיכרון עבור המטריצה כך שתוכל להיות מוצגת עם מימדים חדשים בצורה "רציפה" בזיכרון. זה חשוב לאחר השימוש ב-transpose.
    # מאחדים את התוצאה ממספר הראשים לכדי וקטור embedding אחד.
    # d_out - embeddingוקטור ה
    context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec) # המודל מבצע הקרנה סופית (output projection) על התוצאה
    return context_vec


class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5   # ערך קטן שנוסף לסטיית תקן כדי למנוע חלוקה ב0
    # שכבה ששולטת בכמה למתוח או לכווץ את הערכים המנורמלים, כלומר את הפלטים אחרי הנרמול.
    self.scale = nn.Parameter(torch.ones(emb_dim))
    # שכבה שקובעת בכמה להזיז את הערכים המנורמלים מעלה או מטה אחרי הנרמול.
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift # זה מה שתחזיר את הנורמליזציה


class GELU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    # MultiHeadAttentionהגדרת הפרמטרים בתוך ה
    self.att = MultiHeadAttention(
      d_in=cfg["emb_dim"],
      d_out=cfg["emb_dim"],
      context_length=cfg["context_length"],
      num_heads=cfg["n_heads"],
      dropout=cfg["drop_rate"],
      qkv_bias=cfg["qkv_bias"])
    self.ff = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

  def forward(self, x):
    # Shortcut connection for attention block
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + shortcut    # Add the original input back

    # Shortcut connection for feed forward block
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut    # Add the original input back
    return x


class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    # שכבות הטמעה Embedding שכבות ה
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])   # emb_dim ממפה את אוצר המילים לוקטורים במימד
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])   # שכבת הטמעה למיקומים, שמוסיפה מידע על המיקום של כל טוקן.
    # dropout שכבת ה
    self.drop_emb = nn.Dropout(cfg["drop_rate"])
    # transformer blocksשכבת ה
    self.transformer_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])   # סדרה של בלוקים של טרנספורמרים. בשלב הזה, הבלוקים הם רק תחליפים ויוחלפו בהמשך.
    # שכבת נרמול שמיועדת לשלב הסופי של המודל לפני הפלט.
    self.final_norm = LayerNorm(cfg["emb_dim"])   # שכבת נרמול שתוחלף בהמשך בלוגיקת נרמול אמתית.
    # שבכה אחרונה שמבצעת את המיפוי מהפלט של הבלוקים של הטרנספורמר לפלט הסופי.
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)   # מיפוי הפלט הסופי לערכים שמיצגים את ההסתברות של כל מילה או טוקן בפלט המודל.
    #
    '''זוהי שכבה אחרונה במודל שמבצעת את הפלט הסופי של המודל.
     הפלט של כל הבלוקים של הטרנספורמרים הוא וקטור עם ממד גדול,
      והשכבה הזו ממפה אותו לממד קטן יותר שמתאים למספר האפשרויות הסופיות של המודל
      (כמו מספר המילים באוצר המילים במקרה של מודל GPT).'''

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape   # מגדירה את גודל הבאטצ' (batch) ואת אורך הרצף של הטוקנים.
    tok_embeds = self.tok_emb(in_idx)   # מיפוי הטוקנים לוקטורים בעזרת השכבת הטמעה למעלה
    pos_embeds = self.pos_emb(torch.arange(seq_len))   # הטמעה למיקומים, שמוסיפה מידע על המיקום של כל טוקן. # (torch.arange(seq_len, device=in_idx.device))
    x = tok_embeds + pos_embeds   # Embeddingפלט ה
    x = self.drop_emb(x)   # dropout העברת הפלט דרך שכבת
    x = self.transformer_blocks(x)   # העברת הפלט דרך הבלוקים של הטרנספורמרים.
    x = self.final_norm(x)   # נרמול הפלט בעזרת final_norm.
    logits = self.out_head(x)   # שליחת הפלט לשכבת ה-Layer הסופית שתחשב את הלוגיטים (logits).
    return logits


# שכבה נפרדת מהארכיטקטורה אך חיונית
def generate_text_simple(model, idx, max_new_tokens, context_size):   # idx מערך בגודל (batch_size, n_tokens) המכיל אינדקסים של הטוקנים בהקשר הנוכחי.
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]   # חותכים את ההקשר הנוכחי אם הוא חורג מגודל ההקשר הנתמך. לדוגמה, אם מודל השפה הגדול (LLM) תומך רק ב-5 טוקנים, וגודל ההקשר הוא 10, אז משתמשים רק ב-5 הטוקנים האחרונים כהקשר.
    with torch.no_grad():
      logits = model(idx_cond)

    logits = logits[:, -1, :]   # מתמקדים רק בצעד הזמן האחרון, כך שצורת הטנסור עוברת מ-(batch_size, n_tokens, vocab_size) ל-(batch_size, vocab_size).
    probas = torch.softmax(logits, dim=-1)   # הוא טנסור בצורת (batch_size, vocab_size).
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)   # הוא טנסור בצורת (batch_size, 1).
    idx = torch.cat((idx, idx_next), dim=1) # מצרפים את האינדקס שנבחר לרצף המתמשך, כך ש-idx כעת בצורת (batch_size, n_tokens + 1).
  return idx
