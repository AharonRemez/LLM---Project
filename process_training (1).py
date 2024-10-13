import torch
import torch.nn
from gpt_model import generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):   # For-loop זהה לקודם: קבל לוגיטים, והתמקד רק בשלב האחרון
      idx_cond = idx[:, -context_size:]
      with torch.no_grad():
        logits = model(idx_cond)
      logits = logits[:, -1, :]
      if top_k is not None:   # בחלק החדש הזה, אנו מסננים לוגיטים עם דגימת top_k
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(logits < min_val, torch.tensor(float('-inf')), logits)   # יכול להיות float('-inf')).to(logits.device)

      if temperature > 0.0:   # זהו הסעיף החדש שבו אנו מיישמים קנה מידה של טמפרטורה
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
      else:   # בצע בחירת אסימון הבא חמדנית כמו קודם כאשר קנה המידה של הטמפרטורה מושבת
          idx_next = torch.argmax(logits, dim=-1, keepdim=True)
      if idx_next == eos_id:   # הפסק ליצור מוקדם אם נתקלים באסימון של סוף הרצף וצוין eos_id
        break
      idx = torch.cat((idx, idx_next), dim=1)
    return idx


def calc_loss_batch(input_batch, target_batch, model):   # הוספת device
  input_batch, target_batch = input_batch, target_batch   # יכול להוסיף לכל אחד מהם .to(device)
  logits = model(input_batch)
  loss = torch.nn.functional.cross_entropy(
  logits.flatten(0, 1), target_batch.flatten()
  )
  return loss

def calc_loss_loader(data_loader, model, num_batches=None):   # יכול להוסיף device
  total_loss = 0.
  if len(data_loader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches = len(data_loader)   # batchאיטרטיבי על כל ה
  else:
    num_batches = min(num_batches, len(data_loader))   # הפחת את מספר האצוות כדי להתאים למספר הכולל של אצוות בטעינת הנתונים אם num_batches חורג ממספר האצוות במטען הנתונים
  for i, (input_batch, target_batch) in enumerate(data_loader):
    if i < num_batches:
      loss = calc_loss_batch(input_batch, target_batch, model) # יכול להוסיף device
      total_loss += loss.item()   # Sum loss for each batch
    else:
        break
  return total_loss / num_batches   # Average the loss over all batches


def evaluate_model(model, train_loader, val_loader, eval_iter): # יכול להיות device
  model.eval()   # Dropout is disabled during evaluation for stable, reproducible results
  with torch.no_grad():   # Disable gradient tracking, which is not required during evaluation, to reduce the computational overhead
    train_loss = calc_loss_loader(train_loader, model, num_batches=eval_iter) # יכול להיות device
    val_loss = calc_loss_loader(val_loader, model, num_batches=eval_iter) # יכול להיות device
  model.train()
  return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, start_context): # יכול להיות device
  model.eval()
  context_size = model.pos_emb.weight.shape[0]
  encoded = text_to_token_ids(start_context, tokenizer) # יכול להיות .to(device)
  with torch.no_grad():
    token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) # Compact print format
  model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer): # יכול להיות device
  train_losses, val_losses, track_tokens_seen = [], [], []   # Initialize lists to track losses and tokens seen
  tokens_seen, global_step = 0, -1

  for epoch in range(num_epochs):   # Start the main training loop
    model.train()
    for input_batch, target_batch in train_loader:
      optimizer.zero_grad()   # Reset loss gradients from previous batch iteration
      loss = calc_loss_batch(input_batch, target_batch, model) # יכול להיות device
      loss.backward()   # Calculate loss gradients
      optimizer.step()   # Update model weights using loss gradients
      tokens_seen += input_batch.numel()
      global_step += 1
      if global_step % eval_freq == 0:   # Optional evaluation step
        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, eval_iter) # יכול להיות device
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(tokens_seen)
        print(f"Ep {epoch+1} (Step {global_step:06d}): "
        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

    # Print a sample text after each epoch
    generate_and_print_sample(model, tokenizer, start_context) # יכול להיות device
  return train_losses, val_losses, track_tokens_seen