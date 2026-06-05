# =============================================================================
#  VAE — Vision component (V) του baseline world model
#  Έκδοση για Kaggle Notebook (χωρίς argparse / CLI parsers)
#
#  Τρόπος χρήσης σε notebook:
#    1) Τρέξε πρώτα το cell με το loader.py (ορίζει το VaeDataset)
#       — ή πρόσθεσε το loader.py ως αρχείο/utility script.
#    2) Άλλαξε ό,τι θες στο CFG παρακάτω.
#    3) Τρέξε αυτό το cell· στο τέλος καλείται το load_or_train_vae(CFG):
#         - αν υπάρχει ήδη checkpoint -> το φορτώνει (skip training)
#         - αλλιώς -> εκπαιδεύει και σώζει το ΚΑΛΥΤΕΡΟ μοντέλο.
#
#  ΠΡΟΣΟΧΗ (Kaggle persistence): το /kaggle/working ΑΔΕΙΑΖΕΙ σε κάθε νέο session.
#  Για να επιβιώνει το vae.pth ανάμεσα σε sessions:
#    - Notebook Settings -> Persistence -> "Files only", Ή
#    - κατέβασε το vae.pth και ανέβασέ το ως Kaggle Dataset, και βάλε στο
#      CFG.weights το path τύπου "/kaggle/input/<dataset>/vae.pth".
# =============================================================================
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Το VaeDataset ορίζεται στο loader.py.
#  - Αν το loader.py έχει προστεθεί ως αρχείο/utility script -> γίνεται import.
#  - Αν το έχεις κάνει paste σε προηγούμενο cell, η κλάση υπάρχει ήδη στο
#    global namespace και το import απλώς προσπερνιέται.
try:
    from loader import VaeDataset
except (ImportError, ModuleNotFoundError):
    pass


# ------------------------------- CONFIG --------------------------------------
class CFG:
    """ Ρυθμίσεις εκπαίδευσης (άλλαξέ τες εδώ αντί για command-line arguments). """
    train      = "/kaggle/working/cartpole_data/train"   # φάκελος train (.npz)
    val        = "/kaggle/working/cartpole_data/val"     # φάκελος val
    test       = "/kaggle/working/cartpole_data/test"    # final held-out (ΔΕΝ χρησιμοποιείται εδώ)

    # --- checkpoint paths ---
    save       = "/kaggle/working/vae.pth"   # ΠΟΥ γράφεται το ΚΑΛΥΤΕΡΟ μοντέλο (έξοδος)
    weights    = "/kaggle/working/vae.pth"   # ΑΠΟ ΠΟΥ φορτώνονται έτοιμα βάρη (είσοδος)·
                                             # για επαναχρησιμοποίηση από Kaggle Dataset βάλε
                                             # π.χ. "/kaggle/input/<dataset>/vae.pth"
    force_retrain = False                    # True -> εκπαίδευσε ξανά ακόμη κι αν υπάρχει checkpoint

    # --- training ---
    latent     = 64        # latent size
    shift      = 0         # επίπεδο θορύβου: 0 (clean) | 2 | 5 | 10
    buffer     = 200       # # train αρχεία ανά buffer (RAM) — chunked
    val_buffer = 0         # # val αρχεία ανά buffer· 0 -> ΟΛΟ το val μονομιάς (πλήρες validation)
    beta       = 1.0       # βάρος KL
    batch      = 64
    epochs     = 200       # μέγιστος αριθμός epochs
    patience   = 10        # early stop μετά από τόσα epochs χωρίς βελτίωση στο val
    min_delta  = 0.0       # ελάχιστη βελτίωση val loss για να μετρήσει ως πρόοδος
    lr         = 1e-3      # learning rate (Adam)


class VAE(nn.Module):
    """ Vision component (V) του world model.

    Encoder: 3 conv layers με stride 2 -> υποδιπλασιάζουν 3 φορές (÷8).
    Με είσοδο 80x120 ο encoder βγάζει (64, 10, 15) -> flatten 64*10*15.
    ΠΡΟΣΟΧΗ: αν αλλάξεις ανάλυση εικόνας, άλλαξε και το 64*10*15 παρακάτω.
    """

    def __init__(self, latent_size=64):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_logvar = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_decode = nn.Linear(latent_size, 64 * 10 * 15)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x).reshape(batch_size, -1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.fc_decode(z).view(batch_size, 64, 10, 15)
        reconstructed = self.decoder(decoded)
        return reconstructed, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """ Καθαρό baseline loss: reconstruction (MSE) + beta * KL. """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div


def train_one_chunk(model, train_loader, optimizer, device, beta=1.0):
    """ Μία διέλευση εκπαίδευσης πάνω από το τρέχον buffer (chunk).
    Επιστρέφει (άθροισμα loss, πλήθος δειγμάτων) ώστε να αθροίζεται σωστά
    κατά μήκος πολλών chunks σε ένα epoch. """
    model.train()
    total_loss = 0.0
    for data in tqdm(train_loader, desc="Training Progress", leave=False):
        inputs, _label, _action, _use1, _use2 = data
        x = inputs.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss, len(train_loader.dataset)


@torch.no_grad()
def evaluate(model, val_loader, device, beta=1.0):
    """ Validation σε held-out δεδομένα. Per-sample loss (συγκρίσιμο μεταξύ epochs).

    Με CFG.val_buffer = 0 ο ValLoader έχει φορτώσει ΟΛΟ το val set σε ένα buffer,
    οπότε αυτή η μία διέλευση καλύπτει ολόκληρο το validation set. """
    model.eval()
    total_loss = 0.0
    for data in val_loader:
        inputs, _label, _action, _use1, _use2 = data
        x = inputs.to(device)
        recon_x, mu, logvar = model(x)
        total_loss += vae_loss(recon_x, x, mu, logvar, beta=beta).item()
    return total_loss / max(len(val_loader.dataset), 1)


def train_vae(cfg, model, device):
    """ Πλήρης βρόχος εκπαίδευσης με validation + early stopping.
    Γράφει το ΚΑΛΥΤΕΡΟ μοντέλο στο cfg.save. Επιστρέφει το best val loss. """
    traindataset = VaeDataset(root=cfg.train, shift=cfg.shift, buffer_size=cfg.buffer)
    TrainLoader = DataLoader(traindataset, batch_size=cfg.batch, shuffle=True, drop_last=True)

    valdataset = VaeDataset(root=cfg.val, shift=cfg.shift, buffer_size=cfg.val_buffer)
    ValLoader = DataLoader(valdataset, batch_size=cfg.batch, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # 1 epoch = πλήρης διέλευση ΟΛΩΝ των train chunks
    num_train_chunks = math.ceil(len(traindataset._files) / traindataset._buffer_size)
    # val_buffer=0 -> ολόκληρο το val σε ένα buffer· φορτώνεται ΜΙΑ φορά (σταθερό val set)
    valdataset.load_next_buffer()

    best_val = float('inf')
    epochs_no_improve = 0
    for epoch in range(cfg.epochs):
        epoch_loss, epoch_n = 0.0, 0
        traindataset._buffer_index = 0  # ίδιο chunking σε κάθε epoch (αναπαραγωγιμότητα)
        for _ in range(num_train_chunks):
            traindataset.load_next_buffer()  # επόμενο chunk αρχείων
            chunk_loss, chunk_n = train_one_chunk(model, TrainLoader, optimizer, device, beta=cfg.beta)
            epoch_loss += chunk_loss
            epoch_n += chunk_n
        train_loss = epoch_loss / max(epoch_n, 1)

        val_loss = evaluate(model, ValLoader, device, beta=cfg.beta)
        print(f'Epoch {epoch + 1}: train {train_loss:.4f} | val {val_loss:.4f}')

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg.save)  # κρατάμε το ΚΑΛΥΤΕΡΟ μοντέλο
            print(f'  ✓ νέο καλύτερο val ({best_val:.4f}) -> αποθήκευση στο {cfg.save}')
        else:
            epochs_no_improve += 1
            print(f'  χωρίς βελτίωση ({epochs_no_improve}/{cfg.patience})')
            if epochs_no_improve >= cfg.patience:
                print(f'Early stopping στο epoch {epoch + 1}. Καλύτερο val loss: {best_val:.4f}')
                break

    return best_val


def load_or_train_vae(cfg=CFG):
    """ Επιστρέφει (vae_model, device) έτοιμο για χρήση.

    - Αν υπάρχει checkpoint στο cfg.weights (και force_retrain=False) -> το φορτώνει
      και ΠΑΡΑΛΕΙΠΕΙ την εκπαίδευση.
    - Αλλιώς εκπαιδεύει, σώζει το ΚΑΛΥΤΕΡΟ στο cfg.save και το επαναφορτώνει
      (ώστε το μοντέλο που επιστρέφεται να είναι το best, όχι του τελευταίου epoch).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = VAE(latent_size=cfg.latent).to(device)

    if (not cfg.force_retrain) and os.path.isfile(cfg.weights):
        model.load_state_dict(torch.load(cfg.weights, map_location=device))
        model.eval()
        print(f"[VAE] Φορτώθηκαν έτοιμα βάρη από {cfg.weights} -> παράλειψη εκπαίδευσης.")
        return model, device

    print(f"[VAE] Δεν βρέθηκε checkpoint στο {cfg.weights} (ή force_retrain=True) -> εκπαίδευση.")
    best_val = train_vae(cfg, model, device)
    # επαναφόρτωση του ΚΑΛΥΤΕΡΟΥ checkpoint (early stopping)
    model.load_state_dict(torch.load(cfg.save, map_location=device))
    model.eval()
    print(f"[VAE] Φορτώθηκε το καλύτερο μοντέλο (val={best_val:.4f}) από {cfg.save}.")
    return model, device


# Φορτώνει έτοιμα βάρη αν υπάρχουν, αλλιώς εκπαιδεύει & σώζει το καλύτερο μοντέλο.
vae_model, device = load_or_train_vae(CFG)
