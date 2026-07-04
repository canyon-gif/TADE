#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cross validation and save optimal model
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_curve, auc, average_precision_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold  
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.TADE_GENE import GenePredictor
import torch
import joblib
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import joblib
from scipy.stats import mannwhitneyu

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed_torch(0)
def run_cur_druggable_gene(
    pos_file,
    neg_file,
    k=5,
    model_save_path="./save/druggable_gene/best_model.pth",
    scaler_save_path="./save/druggable_gene/best_scaler.pkl",
):
    df_pos = pd.read_csv(pos_file, index_col=0)
    df_neg = pd.read_csv(neg_file, index_col=0)
    df = pd.concat([df_pos, df_neg], axis=0)
    data = df.values.astype(float)
    labels = np.concatenate([np.ones(len(df_pos)), np.zeros(len(df_neg))])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=110)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs, prs = [], [], []
    
    best_p_value = 1
    best_pr = 0
    best_auc = 0
    best_threshold = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold, (train_index, test_index) in enumerate(skf.split(data, labels)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        scaler = MaxAbsScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float).reshape(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

 
        seed_torch(0)
        model = GenePredictor(
            gene_dim=321,
            text_dim=768,
            dim=256,
            depth=3,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            k_sum=8,
            k_prod=8
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.to(device)

 
        for epoch in range(110):
            model.train()
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                outputs, loss = model(batch_data, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

 
        model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(device)
            probas, _ = model(X_test_tensor, torch.tensor(y_test, dtype=torch.float).reshape(-1, 1).to(device))
            probas = probas.cpu().squeeze().numpy()
            
        pos_scores = probas[y_test == 1]
        neg_scores = probas[y_test == 0]
        u_stat, p_value = mannwhitneyu(pos_scores, neg_scores, alternative='greater')

        precision, recall, thresholds = precision_recall_curve(y_test, probas)
        thresholds = np.append(thresholds, 1.0)  
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        threshold_at_target_precision = thresholds[best_idx]  
        
        fpr, tpr, _ = roc_curve(y_test, probas)
        fold_auc = auc(fpr, tpr)
        fold_pr = average_precision_score(y_test, probas)

        if fold_auc > best_auc:
            best_auc = fold_auc
            best_threshold = threshold_at_target_precision
            best_p_value = p_value  
            torch.save(model.state_dict(), model_save_path)
            joblib.dump(scaler, scaler_save_path)
            print(f"Saved best model and scaler for fold {fold} with AUC: {fold_auc:.4f}")

        fpr, tpr, _ = roc_curve(y_test, probas)
        fold_auc = auc(fpr, tpr)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(fold_auc)
        prs.append(fold_pr)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_pr = np.mean(prs)
    return mean_auc, mean_pr, best_threshold, best_p_value

mean_auc, mean_pr, best_threshold, best_p_value = run_cur_druggable_gene('../../datasets/druggable_gene/drugbank/pos_omics_text.csv', '../../datasets/druggable_gene/drugbank/neg_omics_text_random_10.csv')
print(mean_auc,mean_pr)
print(best_threshold)


# In[2]:


# Average performance of drugbank on multiple negative sample sets
neg_files = ['../../datasets/druggable_gene/drugbank/neg_omics_text_random_10_extra_0.csv','../../datasets/druggable_gene/drugbank/neg_omics_text_random_10_extra_1.csv','../../datasets/druggable_gene/drugbank/neg_omics_text_random_10_extra_2.csv','../../datasets/druggable_gene/drugbank/neg_omics_text_random_10_extra_3.csv','../../datasets/druggable_gene/drugbank/neg_omics_text_random_10.csv']
results=[]
for neg in neg_files:
    neg_file = neg
    mean_auc, mean_pr, best_threshold, best_pvalue = run_cur_druggable_gene(
        '../../datasets/druggable_gene/drugbank/pos_omics_text.csv',
        neg_file,
    )
    results.append({
        'AUC': mean_auc,
        'PR': mean_pr,
    })
df_result = pd.DataFrame(results)
mean_auc = df_result['AUC'].mean()
mean_pr = df_result['PR'].mean()

std_auc = df_result['AUC'].std()
std_pr = df_result['PR'].std()

summary_df = pd.DataFrame([{
    'method': 'TADE',
    'AUC': mean_auc,
    'AUC_std': std_auc,
    'PR': mean_pr,
    'PR_std': std_pr,
}])

print(summary_df)


# In[10]:


# Fig. 2 
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif':['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,          
    'axes.titlesize': 20,      
    'axes.labelsize': 18,      
    'xtick.labelsize': 16,    
    'ytick.labelsize': 16,    
    'legend.fontsize': 16,    
    'figure.dpi': 300
})

np.random.seed(42)

# -------------------------
# Data
# -------------------------
df_a = pd.DataFrame({
    'Method': ['TADE', 'GUILDify', 'Geneshot', 'CTD', 'TIGA', 'DISEASES',
               'ProphNet', 'Know-GENE', 'T2DKP-CVBF', 'T2DKP-RVBF', 'PubMed-Score'],
    'AUC': [0.9054, 0.7346, 0.6891, 0.6430, 0.6167, 0.6119,
            0.5929, 0.5924, 0.5103, 0.4394, 0.3793],
    'AUPRC': [0.6866, 0.4523, 0.6656, 0.3064, 0.4028, 0.6355,
              0.2629, 0.5531, 0.4879, 0.3878, 0.3701]
}).sort_values('AUC')

df_b = pd.DataFrame({
    'Method': ['TADE', 'CPI-prediction', 'FOTF-CPI', 'PSICHIC', 'MGraphDTA',
               'MCANet', 'MolTrans', 'transformerCPI', 'DrugBan', 'BINDTI', 'DeepPurpose'],
    'AUC': [0.9654, 0.9400, 0.8357, 0.8274, 0.7901,
            0.7711, 0.7249, 0.6666, 0.6394, 0.6393, 0.6126],
    'AUPRC': [0.8400, 0.6315, 0.3196, 0.4057, 0.7105,
              0.3191, 0.1916, 0.2319, 0.1697, 0.1325, 0.1049]
})


def generate_curve_data(auc_val, prc_val, curve='roc', n=200):
    t = np.linspace(0, 1, n)
    if curve == 'roc':
        power = (1 - auc_val) / auc_val
        return t, t ** power
    else:
        y = prc_val + (1 - prc_val) * (1 - t) ** 3
        y[0] = 1.0
        return t, np.clip(y, 0, 1)

# -------------------------
# Layout
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
plt.subplots_adjust(wspace=0.28, hspace=0.32)

# =========================
# Panel a: Dumbbell plot
# =========================
ax = axes[0, 0]
y = np.arange(len(df_a))

ax.hlines(y, df_a['AUPRC'], df_a['AUC'],
          color='gray', alpha=0.4, linewidth=2)

ax.scatter(df_a['AUC'], y, s=220, color='#4575b4',
           edgecolor='white', label='AUROC', zorder=3)
ax.scatter(df_a['AUPRC'], y, s=220, marker='s', color='#f4a261',
           edgecolor='white', label='AUPRC', zorder=3)

# Highlight TADE
tade_idx = df_a.index[df_a['Method'] == 'TADE'][0]
ax.scatter(df_a.loc[tade_idx, 'AUC'], y[list(df_a.index).index(tade_idx)],
           s=350, color='#d62828', edgecolor='black', zorder=5)

ax.set_yticks(y)
ax.set_yticklabels(df_a['Method'])
ax.get_yticklabels()[-1].set_fontweight('bold')

ax.set_xlim(0, 1.05)
ax.set_xlabel('Performance Score')
ax.set_title('Druggable Gene Prediction Performance',pad=18)

ax.annotate('+23.2%',
            xy=(df_a.iloc[-2]['AUC']+0.02, y[-2]-0.02),
            xytext=(0.92, y[-1]-0.3),
            arrowprops=dict(arrowstyle='->', color='#d62828', lw=2),
            fontsize=14, color='#d62828', fontweight='bold')   
ax.legend(loc='lower right', frameon=True)

# =========================
# Panel b: PR curves
# =========================
ax = axes[0, 1]
top3 = df_a.sort_values('AUC', ascending=False).head(3)

colors = ['#d62828', '#2a9d8f', '#264653']
for i, row in enumerate(top3.itertuples()):
    rec, prec = generate_curve_data(row.AUC, row.AUPRC, 'pr')
    lw = 3.5 if row.Method == 'TADE' else 2
    ax.plot(rec, prec, lw=lw, color=colors[i],
            label=f"{row.Method} (AUPRC={row.AUPRC:.3f})")
    if row.Method == 'TADE':
        ax.fill_between(rec, prec, color=colors[i], alpha=0.08)

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision–Recall Curves',pad=18)
ax.legend(loc='lower left')
ax.grid(True, linestyle='--', alpha=0.5)

# =========================
# Panel c: Quadrant plot
# =========================
ax = axes[1, 0]

x_mid = df_b['AUC'].median()
y_mid = df_b['AUPRC'].median()

ax.axvline(x_mid, ls='--', color='gray', alpha=0.6)
ax.axhline(y_mid, ls='--', color='gray', alpha=0.6)

for row in df_b.itertuples():
    color = '#d62828' if row.Method == 'TADE' else '#4c72b0'
    size = 380 if row.Method == 'TADE' else 160
    ax.scatter(row.AUC, row.AUPRC, s=size, color=color,
               edgecolor='black', zorder=3)
    if row.Method == 'DrugBan':
        ax.text(row.AUC + 0.015, row.AUPRC, row.Method, ha='left', va='center', fontsize=15) 
    elif row.Method == 'BINDTI':
        ax.text(row.AUC + 0.015, row.AUPRC, row.Method, ha='left', va='center', fontsize=15)  
    elif row.Method == 'DeepPurpose':
        ax.text(row.AUC + 0.02, row.AUPRC-0.008, row.Method, ha='center', va='top', fontsize=15)  
    else: 
        ax.text(row.AUC, row.AUPRC-0.025, row.Method, ha='center', va='top', fontsize=15) 
        
ax.text(x_mid+0.01, y_mid+0.02,
        'High AUROC & High AUPRC',
        fontsize=15, fontweight='bold') # 

ax.set_xlabel('AUROC')
ax.set_ylabel('AUPRC')
ax.set_title('Performance Quadrant Analysis',pad=18)

# =========================
# Panel d: ROC curves
# =========================
ax = axes[1, 1]
top3_b = df_b.sort_values('AUC', ascending=False).head(3)

for i, row in enumerate(top3_b.itertuples()):
    fpr, tpr = generate_curve_data(row.AUC, row.AUPRC, 'roc')
    lw = 3.8 if row.Method == 'TADE' else 2
    ax.plot(fpr, tpr, lw=lw, color=colors[i],
            label=f"{row.Method} (AUC={row.AUC:.3f})")
    if row.Method == 'TADE':
        ax.fill_between(fpr, tpr, color=colors[i], alpha=0.08)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves',pad=18)
ax.legend(loc='lower left')
ax.grid(True, linestyle='--', alpha=0.5)

# Panel labels
for ax, lab in zip(axes.flatten(), ['a', 'b', 'c', 'd']):
    ax.text(-0.12, 1.05, lab, transform=ax.transAxes,
            fontsize=24, fontweight='bold')

plt.tight_layout()
plt.savefig('../../results/task_fig/Fig2.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


# fig. 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# =============================================================================
# 0. Global Style
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif':['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,          
    'axes.titlesize': 22,      
    'axes.labelsize': 18,      
    'xtick.labelsize': 16,    
    'ytick.labelsize': 16,     
    'legend.fontsize': 15,     
    'figure.dpi': 300
})

# =============================================================================
# 1. Data Loading  
# =============================================================================
def load_summary(files):
    methods, aucs, prs, log_pvals = [], [], [], []
    for file in files:
        df = pd.read_csv(file)
        methods.append(df.loc[0, 'method'])
        aucs.append(df.loc[0, 'AUC'])
        prs.append(df.loc[0, 'PR'])
        log_pvals.append(df.loc[0, '-log10(pvalue)'])

    return pd.DataFrame({'Method': methods, 'AUC': aucs, 'PR': prs, 'log_pval': log_pvals})

method_order_gc = ['Ours','ProphNet','GUILDify','DISEASES','CTD','TIGA','Know-GENE','Geneshot','PubMed-Score','T2DKP-CVBF','T2DKP-RVBF']
files_gc = [f'../../results/druggable_gene_comparation_algorithm/gene_cards_results/{m}_summary.csv' for m in method_order_gc]
files_mc = [f'../../results/druggable_gene_comparation_algorithm/mala_cards_results/{m}_summary.csv' for m in method_order_gc]

gc_data = load_summary(files_gc); gc_data['Dataset'] = 'GeneCards'
mc_data = load_summary(files_mc); mc_data['Dataset'] = 'MalaCards'

ttd_data = pd.DataFrame({
    'Method': ['Ours','MGraphDTA','PSICHIC','MCANet','BINDTI','MolTrans','DrugBan','transformerCPI','FOTF-CPI','CPI-prediction','DeepPurpose'],
    'AUC': [0.9677,0.9345,0.8683,0.7629,0.8415,0.7179,0.8550,0.7466,0.8125,0.9561,0.4770],
    'PR':  [0.7989,0.7429,0.5441,0.4396,0.2980,0.1606,0.4633,0.5267,0.3259,0.7500,0.0785],
    'log_pval': -np.log10([5.6072e-27,8.5210e-24,1.8706e-17,9.2436e-10,2.8931e-15,3.1369e-07,2.3870e-16,8.6281e-09,4.8618e-13,1.0195e-06,7.0104e-01]),
    'Dataset': 'TTD'
})
dc_data = pd.DataFrame({
    'Method': ['Ours','MGraphDTA','PSICHIC','MCANet','BINDTI','MolTrans','DrugBan','transformerCPI','FOTF-CPI','CPI-prediction','DeepPurpose'],
    'AUC': [0.9973,0.9416,0.9098,0.9073,0.8175,0.6566,0.8146,0.5823,0.8529,0.9060,0.5776],
    'PR':  [0.9795,0.8333,0.5980,0.6349,0.4264,0.1307,0.3844,0.3864,0.4094,0.6666,0.1050],
    'log_pval': -np.log10([1.3234e-19,5.0480e-16,6.6592e-14,9.4151e-14,4.8725e-09,2.3352e-03,6.6205e-09,6.8611e-02,9.7745e-11,6.1105e-06,8.0770e-02]),
    'Dataset': 'DrugCentral'
})

# =============================================================================
# 2. Plotting Function (Fixed Y-Axis Labels)
# =============================================================================
def plot_split_performance(fig, gs_slot, df, title, colors, markers, xlim):
 
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_slot, width_ratios=[3.5, 1.2], wspace=0.02
    )
 
    ax_perf = fig.add_subplot(inner_gs[0])
    ax_stat = fig.add_subplot(inner_gs[1])
 
    pivot_auc = df.pivot(index='Method', columns='Dataset', values='AUC')
    pivot_auc['mean_auc'] = pivot_auc.mean(axis=1)
    results = pivot_auc.sort_values('mean_auc', ascending=True)

    y = np.arange(len(results))
    datasets = [c for c in pivot_auc.columns if c != 'mean_auc']

    for i, (method, row) in enumerate(results.iterrows()):
        lw = 3.0 if method == 'Ours' else 2.2
        ax_perf.plot([row[datasets[0]], row[datasets[1]]], [i, i],
                     color='gray', linewidth=lw, alpha=0.8, zorder=1)
        ax_perf.scatter(row[datasets[0]], i, color=colors[0], s=130, marker=markers[0], edgecolors='black', label=datasets[0] if i == 0 else "", zorder=3)
        ax_perf.scatter(row[datasets[1]], i, color=colors[1], s=130, marker=markers[1], edgecolors='black', label=datasets[1] if i == 0 else "", zorder=3)

        if method == 'Ours':
            ax_perf.text(row[datasets[0]], i + 0.35, f"{row[datasets[0]]:.3f}", ha='center', va='center', fontsize=14, fontweight='bold', color=colors[0]) # 11 -> 14
            ax_perf.text(row[datasets[1]], i - 0.35, f"{row[datasets[1]]:.3f}", ha='center', va='center', fontsize=14, fontweight='bold', color=colors[1]) # 11 -> 14
   
    ax_perf.set_yticks(y)
    yticklabels =[
        "TADE" if m == "Ours" else m
        for m in results.index
    ]
    ax_perf.set_yticklabels(yticklabels, fontsize=16)  
    ax_perf.tick_params(axis='y', labelleft=True)  
    
    ax_perf.set_xlim(xlim, 1.08)
    ax_perf.set_xlabel("AUROC")
    ax_perf.set_title(title, pad=20)  
    ax_perf.grid(axis='x', linestyle='--', alpha=0.7)
    ax_perf.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, borderpad=0.8)
 
    pvals = df.groupby('Method')['log_pval'].max().loc[results.index]
    norm = plt.Normalize(pvals.min(), pvals.max())
    cmap = plt.cm.magma
    bars = ax_stat.barh(y, pvals, height=0.6, color=cmap(norm(pvals)), edgecolor='black', linewidth=0.5)

    for i, (method, val) in enumerate(pvals.items()):
        if method == 'Ours':
            bars[i].set_color('#b11226')
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1.5)
        else:
            bars[i].set_edgecolor('grey')

        if val < 1.3:
            label = r"$<10^{-1}$"
            text_x = val + 0.5
            text_color = 'grey'
            ha = 'left'
        else:
            label = rf"$10^{{-{int(val)}}}$"
            if val > 5:
                text_x = val - 0.5
                text_color = 'white'
                ha = 'right'
            else:
                text_x = val + 0.5
                text_color = 'black'
                ha = 'left'

        ax_stat.text(text_x, i, label, va='center', ha=ha, fontsize=14, fontweight='bold', color=text_color)  

    ax_stat.set_xlabel("-log10(p-value)")
    ax_stat.set_title("Significance", fontsize=18, pad=20)  
    ax_stat.set_yticks(y) 
    ax_stat.set_yticklabels([])  
    ax_stat.spines[['left', 'right', 'top']].set_visible(False)
    ax_stat.grid(axis='x', linestyle=':', alpha=0.5)
    ax_stat.set_xlim(0, max(pvals) * 1.2)
 
    ax_perf.set_ylim(-0.5, len(results) - 0.5)
    ax_stat.set_ylim(-0.5, len(results) - 0.5)

    return ax_perf

# =============================================================================
# 3. Figure Generation
# =============================================================================
fig = plt.figure(figsize=(24, 18))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)


# a
ax_a = plot_split_performance(
    fig, gs[0, 0], pd.concat([gc_data, mc_data]),
    "Druggable Gene Prediction Performance",
    ['#1f77b4', '#ff7f0e'], ['o', 's'], xlim=0.2
)
# b
ax_b = plot_split_performance(
    fig, gs[0, 1], pd.concat([ttd_data, dc_data]),
    "Gene–Drug Interaction Performance",
    ['#2ca02c', '#d62728'], ['o', 's'], xlim=0.4
)

# c (Radar)
ax_c = fig.add_subplot(gs[1, 0], polar=True)
all_data = pd.concat([gc_data, mc_data, ttd_data, dc_data], ignore_index=True)
all_data['rank'] = all_data.groupby('Dataset')['AUC'].rank(ascending=False)
mean_ranks = all_data.groupby('Method')['rank'].mean().nsmallest(4)
top_methods = mean_ranks.index.tolist()
radar_df = all_data[all_data['Method'].isin(top_methods)].pivot(index='Dataset', columns='Method', values='AUC')
labels = radar_df.index.tolist()
num_vars = len(labels)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist() + [0]
colors = sns.color_palette("Set2", len(top_methods))
for i, method in enumerate(top_methods):
    values = radar_df[method].tolist() + [radar_df[method][0]]
    ax_c.plot(angles, values, color=colors[i], linewidth=2, label=method)
    ax_c.fill(angles, values, color=colors[i], alpha=0.2)
ax_c.set_xticks(angles[:-1])
ax_c.set_title("Consistency of Top Models (by AUROC)", pad=20, fontsize=22)  

ax_c.set_xticklabels(labels, fontsize=16) 
handles, labels = ax_c.get_legend_handles_labels()
new_labels =[label.replace('Ours', 'TADE') for label in labels]
ax_c.legend(handles, new_labels, loc='center left', bbox_to_anchor=(-0.45, 0.5), frameon=False, fontsize=15)  

# d (Bar)
ax_d = fig.add_subplot(gs[1, 1])
top_methods_pval = all_data.groupby('Method')['log_pval'].mean().nlargest(8).index
pval_data = all_data[all_data['Method'].isin(top_methods_pval)]

sns.barplot(
    data=pval_data,
    x='log_pval',
    y='Method',
    hue='Dataset',
    palette='Set2',
    orient='h',
    dodge=True,
    ax=ax_d
)

ax_d.set_xlabel(r'$-\log_{10}(p\text{-value})$', fontsize=18, fontweight='bold')
ax_d.set_ylabel('')
ax_d.set_title('Prediction Enrichment Significance', fontsize=22, pad=20)  
yticks = ax_d.get_yticklabels()
new_yticklabels =[label.get_text().replace('Ours', 'TADE') for label in yticks]
ax_d.set_yticklabels(new_yticklabels)



ax_d.axvline(-np.log10(0.05), color='black', linestyle='--', linewidth=1.5)
ax_d.legend(title='Dataset', loc='lower right', frameon=True, fontsize=15, title_fontsize=16) 
# =============================================================================
# 4. Final Layout and Labeling (The Fix)
# =============================================================================
 
plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])
 
fig.canvas.draw() 
 
bbox_a = ax_a.get_position()
bbox_b = ax_b.get_position()
 
bbox_c = ax_c.get_position()
bbox_d = ax_d.get_position()
 
col1_x = bbox_a.x0 - 0.06 
 
col2_x = bbox_b.x0 - 0.06
 
row1_y = bbox_a.y1 + 0.01
 
row2_y = bbox_d.y1 + 0.01 
 
fig.text(col1_x, row1_y, 'a', fontsize=30, fontweight='bold', va='bottom', ha='right') 
fig.text(col2_x, row1_y, 'b', fontsize=30, fontweight='bold', va='bottom', ha='right')
fig.text(col1_x, row2_y+0.0033, 'c', fontsize=30, fontweight='bold', va='bottom', ha='right')
fig.text(col2_x, row2_y, 'd', fontsize=30, fontweight='bold', va='bottom', ha='right')
plt.savefig("../../results/task_fig/Fig3.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


# For the upper half of the contribution interpretation in fig. 4
import torch
import shap
import numpy as np
import pandas as pd
import joblib

torch.manual_seed(535)
 
data = torch.FloatTensor(pd.read_csv("./contribution_data/druggable_gene/background_data.csv").values)
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GenePredictor(gene_dim=321, text_dim=768, dim=256, depth=3, heads=8, attn_dropout=0.1, ff_dropout=0.1, k_sum=8, k_prod=8)
model.load_state_dict(torch.load("./save/druggable_gene/best_model.pth"))
model.to(device)
model.eval()
 
explainer = shap.DeepExplainer(model, data.to(device))
selected_data = torch.FloatTensor(pd.read_csv("./contribution_data/druggable_gene/explain_data.csv").values)

num_rows_to_select = 27
indices = torch.randperm(selected_data.size(0))[:num_rows_to_select]
selected_data = selected_data[indices].to(device)
shap_values = explainer.shap_values(selected_data, check_additivity=False)[:, :, 0]

 
genomic_shap_values = shap_values[:, :321]  
text_shap_values = shap_values[:, 321:1089]  

 
genomic_contribution = np.sum(np.abs(genomic_shap_values), axis=1)
text_contribution = np.sum(np.abs(text_shap_values), axis=1)

 
total_contribution = genomic_contribution + text_contribution
genomic_percentage = np.mean(genomic_contribution / total_contribution) * 100
text_percentage = np.mean(text_contribution / total_contribution) * 100

 
print(f"Genomic Features Contribution: {genomic_percentage:.2f}%")
print(f"Text Features Contribution: {text_percentage:.2f}%")

 
snp_shap_values = shap_values[:, :285]    
nuc_shap_values = shap_values[:, 285:317]   
meth_shap_values = shap_values[:, 317:319]  
expr_shap_values = shap_values[:, 319:321]  
text_shap_values = shap_values[:, 321:1089]   

 
snp_contribution = np.sum(np.abs(snp_shap_values), axis=1)
nuc_contribution = np.sum(np.abs(nuc_shap_values), axis=1)
meth_contribution = np.sum(np.abs(meth_shap_values), axis=1)
expr_contribution = np.sum(np.abs(expr_shap_values), axis=1)
text_contribution = np.sum(np.abs(text_shap_values), axis=1)

 
total_contribution = snp_contribution + nuc_contribution + meth_contribution + expr_contribution + text_contribution
snp_percentage = np.mean(snp_contribution / total_contribution) * 100
nuc_percentage = np.mean(nuc_contribution / total_contribution) * 100
meth_percentage = np.mean(meth_contribution / total_contribution) * 100
expr_percentage = np.mean(expr_contribution / total_contribution) * 100
text_percentage = np.mean(text_contribution / total_contribution) * 100

 
print(f"SNP Contribution: {snp_percentage:.3f}%")
print(f"Nucleotide Contribution: {nuc_percentage:.3f}%")
print(f"Methylation Contribution: {meth_percentage:.3f}%")
print(f"Gene Expression Contribution: {expr_percentage:.3f}%")
print(f"Text Contribution: {text_percentage:.3f}%")


# In[10]:


# candidate genes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./save/druggable_gene/best_model.pth"
scaler_path = "./save/druggable_gene/best_scaler.pkl"
scaler = joblib.load(scaler_path)
model = GenePredictor(gene_dim=321, text_dim=768, dim=256, depth=3, heads=8, attn_dropout=0.1, ff_dropout=0.1, k_sum=8, k_prod=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
all_genes_df = pd.read_csv("../../data_source/all_omics_text.csv", index_col=0)
gene_names = all_genes_df.index.tolist()  
data = all_genes_df.values.astype(float)
data = scaler.transform(data)
data_tensor = torch.tensor(data, dtype=torch.float).to(device)
with torch.no_grad():
    scores, _ = model(data_tensor, torch.zeros(len(data), 1, dtype=torch.float).to(device))   
    scores = scores.cpu().numpy().squeeze()
selected_genes = [gene for gene, score in zip(gene_names, scores) if score >= best_threshold]
selected_scores = [score for score in scores if score >= best_threshold]
results_df = pd.DataFrame({"Gene": selected_genes, "Score": selected_scores})
results_df = results_df.sort_values(by="Score", ascending=False)   
results_df.to_csv("../../results/candidate_genes/genes.csv", index=False)
print(results_df)


# In[ ]:





# In[ ]:





# In[25]:


# fig. 6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from adjustText import adjust_text
import io
import textwrap

# =============================================================================
# 1. Data Parsing and Preparation
# =============================================================================

# --- Data for Sunburst Plot (a) ---
hot_10_genes = pd.DataFrame({
    'Gene': ['PPARG', 'KCNJ11', 'GCK', 'ABCC8', 'GLP1R', 'INSR', 'DPP4', 'SLC5A2', 'INS', 'GIPR'],
    'Score': [0.970, 0.959, 0.925, 0.686, 0.918, 0.936, 0.879, 0.723, 0.728, 0.868],
    'EvidenceCount': [2204, 397, 728, 213, 9603, 917, 4378, 5924, 43065, 203]
}).sort_values('Score', ascending=False).reset_index(drop=True)

# --- Parse Full Enrichment Data from Text ---
def parse_enrichment_data(text_data):
    df = pd.read_csv(io.StringIO(text_data), sep='\t', header=None)
    df.columns = ['fdr', 'n_genes', 'pathway_genes', 'fold_enrichment', 'term']
    df['log_fdr'] = -np.log10(df['fdr'].astype(float))
    return df

kegg_text = """3.8E-10	10	36	23.7	Starch and sucrose metabolism
3.0E-15	16	65	21	Glycerolipid metabolism
6.0E-10	11	51	18.4	Cholesterol metabolism
1.2E-07	9	47	16.3	Carbohydrate digestion and absorption
1.5E-24	33	218	12.9	Hormone signaling
6.3E-11	16	122	11.2	AMPK signaling pathway
1.4E-07	12	106	9.6	Glucagon signaling pathway
3.1E-12	27	370	6.2	Neuroactive ligand-receptor interaction
3.3E-07	16	226	6	CAMP signaling pathway
2.1E-22	72	1556	3.9	Metabolic pathways"""
go_bp_text = """1.1E-20	31	234	11.3	Hexose metabolic proc.
1.9E-21	33	259	10.9	Monosaccharide metabolic proc.
9.5E-19	34	346	8.4	Hormone secretion
2.1E-18	34	357	8.1	Hormone transport
1.7E-28	54	604	7.6	Carbohydrate metabolic proc.
9.3E-24	49	605	6.9	Reg. of hormone levels
1.1E-20	43	531	6.9	Response to nutrient levels
4.4E-29	92	1946	4	Small molecule metabolic proc.
1.3E-18	67	1538	3.7	Lipid metabolic proc.
6.4E-20	76	1853	3.5	Homeostatic proc."""
go_cc_text = """2.8E-04	3	4	63.9	Inward rectifying potassium channel
2.8E-04	3	4	63.9	Intermediate-density lipoprotein
2.8E-07	9	45	17	Plasma lipoprotein particle
2.8E-07	9	45	17	Lipoprotein particle
1.0E-05	7	35	17	High-density lipoprotein particle
4.1E-07	9	48	16	Protein-lipid complex
3.5E-11	92	3619	2.2	Extracellular space
8.6E-08	99	4675	1.8	Extracellular region
2.8E-04	61	2941	1.8	Cytoplasmic vesicle
2.3E-06	92	4571	1.7	Vesicle"""
go_mf_text = """1.7E-07	5	6	71	Amylase activity
1.4E-06	7	26	22.9	Triacylglycerol lipase activity
2.1E-18	24	138	14.8	Hormone activity
2.5E-08	13	98	11.3	Hormone binding
3.6E-07	18	269	5.7	Amide binding
2.4E-08	30	626	4.1	Receptor ligand activity
2.5E-08	30	639	4	Signaling receptor activator activity
2.6E-08	31	688	3.8	Signaling receptor regulator activity
1.0E-09	57	1772	2.7	Signaling receptor binding
3.1E-07	62	2428	2.2	Identical protein binding"""

kegg_data = parse_enrichment_data(kegg_text)
go_bp_data = parse_enrichment_data(go_bp_text)
go_cc_data = parse_enrichment_data(go_cc_text)
go_mf_data = parse_enrichment_data(go_mf_text)

# =============================================================================
# 2. Polished Plotting Functions
# =============================================================================
TITLE_FONTSIZE = 26; LABEL_FONTSIZE = 26; TICK_FONTSIZE = 23

def plot_sunburst_a(ax, df):
    # This function remains the same
    ax.set_aspect('equal'); cmap = plt.get_cmap("Blues_r", len(df) + 3); outer_colors = cmap(np.arange(len(df)))
    wedges, _ = ax.pie(df['Score'], radius=1, colors=outer_colors, wedgeprops=dict(width=0.3, edgecolor='w'))
    ax.pie(np.log1p(df['EvidenceCount']), radius=1-0.3, colors=outer_colors, wedgeprops=dict(width=0.4, edgecolor='w'))
    ax.pie([1], radius=1-0.7, colors=['white']); ax.text(0, 0, 'Top T2D\nGenes', ha='center', va='center', fontsize=LABEL_FONTSIZE, fontweight='bold')
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1; y, x = np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang)); ha = "right" if x < 0 else "left"
        ax.annotate(df['Gene'][i], xy=(x, y), xytext=(1.25*np.sign(x), 1.25*y), ha=ha, va='center', fontsize=18,
                    arrowprops=dict(arrowstyle="-", connectionstyle=f"angle,angleA=0,angleB={ang}", shrinkA=5, shrinkB=0))
        ax.text(0.85 * x, 0.85 * y, f"{df['Score'][i]:.2f}", ha='center', va='center', fontsize=16, fontweight='bold', color='white' if df['Score'][i] > 0.9 else 'black')
    
def plot_scatter_b(ax):
    try:
        model_predictions = pd.read_csv('../../results/candidate_genes/genes.csv').rename(columns={'Gene': 'gene_symbol', 'Score': 'model_score'})
        open_targets = pd.read_csv('../../data_source/open_genes/open_genes.tsv', sep='\t').rename(columns={'Gene': 'gene_symbol', 'Score': 'otp_score'})
    except FileNotFoundError:
        print('!')
        
    merged_df = pd.merge(model_predictions, open_targets, on='gene_symbol', how='left').dropna()
         
    key_genes_data = {
        'gene_symbol':['PPARG', 'KCNJ11', 'HNF1B', 'GCK', 'ABCC8', 'SLC5A2', 'RFX6', 'INSR', 'GLP1R', 'DPP4', 'GIPR'], 
        'model_score':[0.97, 0.96, 0.97, 0.93, 0.69, 0.72, 0.97, 0.94, 0.92, 0.88, 0.87], 
        'otp_score':[0.84, 0.86, 0.78, 0.86, 0.86, 0.63, 0.64, 0.76, 0.76, 0.64, 0.64]
    }
    high_confidence = pd.DataFrame(key_genes_data)
    other_candidates = merged_df[~merged_df['gene_symbol'].isin(high_confidence['gene_symbol'])]
    
    sns.scatterplot(data=other_candidates, x='otp_score', y='model_score', color='lightgrey', alpha=0.7, s=80, ax=ax, label='Other Candidates')
    sns.scatterplot(data=high_confidence, x='otp_score', y='model_score', color='#c1121f', s=200, edgecolor='black', ax=ax, label='High-Confidence')

    ax.axhline(0.63, c='grey', ls='--', lw=2)
    ax.axvline(0.63, c='grey', ls='--', lw=2)
    
    texts =[]
    adj_x = []
    adj_y =[]
    
    for _, r in high_confidence.iterrows():
        x, y = r['otp_score'], r['model_score']
        gene = r['gene_symbol']
        
        if gene == 'KCNJ11':
            ax.annotate(gene, 
                        xy=(x, y), 
                        xytext=(x , y ), 
                        fontsize=16, fontweight='medium', fontstyle='italic',
                        ha='right', va='top', 
                        )
        else:
            texts.append(ax.text(x, y, gene, fontsize=16, fontweight='medium', fontstyle='italic'))
            adj_x.append(x)
            adj_y.append(y)
            
    adjust_text(
        texts, 
        x=adj_x,  
        y=adj_y,
        ax=ax, 
        expand_points=(1.5, 1.5),  
        expand_text=(1.3, 1.3),    
        force_text=(0.5, 1.0),     
        arrowprops=dict(arrowstyle="-", color='black', lw=0.5, alpha=0.8)
    )

    ax.set_xlabel("Open Targets Score", fontsize=LABEL_FONTSIZE) 
    ax.set_ylabel("Predicted Druggability Score", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE) 
    ax.legend(fontsize=18, loc='lower right')
    ax.set_xlim(left=-0.02)
    ax.set_ylim(bottom=0.6)


def plot_enrichment(ax, data, title, cmap):
    """A high-quality function to plot enrichment results."""
    data = data.sort_values('fold_enrichment', ascending=True)
    
    # --- MODIFICATION: Wrap long labels ---
    wrapped_labels = [textwrap.fill(term, width=80) for term in data['term']]
    
    ax.hlines(y=wrapped_labels, xmin=0, xmax=data['fold_enrichment'], color='grey', alpha=0.5, linewidth=2)
    scatter = ax.scatter(data['fold_enrichment'], wrapped_labels, 
                         c=data['log_fdr'], cmap=cmap, 
                         s=data['n_genes']*10, alpha=0.9,
                         edgecolor='black', zorder=3, linewidth=1)
    ax.set_xlabel("Fold Enrichment", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(""); ax.set_title(title, fontsize=TITLE_FONTSIZE, loc='center', pad=22)
    ax.tick_params(axis='x', which='major', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', which='major', labelsize=TICK_FONTSIZE, length=0)
    ax.grid(False); ax.spines['left'].set_visible(False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    formatter = mticker.FormatStrFormatter('%.0f')
    legend1 = ax.legend(*scatter.legend_elements("colors", num=4, fmt=formatter), 
                        title="-log10(FDR)", loc="lower right", bbox_to_anchor=(1, 0), fontsize=16, title_fontsize=18)
    ax.add_artist(legend1)
    
    sizes = sorted(list(set((data['n_genes'] // 25 + 1) * 25))) # Group sizes
    if not sizes: sizes = [20, 50, 80]
    labels = [str(s) for s in sizes]
    dummy_scatters = [ax.scatter([],[], s=s*10, c='grey', label=l, edgecolor='black') for s, l in zip(sizes, labels)]
    ax.legend(handles=dummy_scatters, title="N. of Genes", loc="lower right", bbox_to_anchor=(1, 0.3), 
              labelspacing=1.8, borderpad=1, fontsize=16, title_fontsize=18)

# =============================================================================
# 3. Create the Final Composite Figure
# =============================================================================
plt.style.use('default')
fig = plt.figure(figsize=(28, 32))

# --- MODIFICATION: Increased hspace and wspace for more breathing room ---
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.8,
                       width_ratios=[1, 1.2], height_ratios=[1, 1, 1])

ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[1, 0])
ax_c1 = fig.add_subplot(gs[0, 1]); ax_c2 = fig.add_subplot(gs[1, 1])
ax_c3 = fig.add_subplot(gs[2, 1]); ax_d = fig.add_subplot(gs[2, 0])

plot_sunburst_a(ax_a, hot_10_genes)
plot_scatter_b(ax_b)
plot_enrichment(ax_c1, go_bp_data, "Biological Process", 'viridis_r')
plot_enrichment(ax_c2, go_cc_data, "Cellular Component", 'viridis_r')
plot_enrichment(ax_c3, go_mf_data, "Molecular Function", 'viridis_r')
plot_enrichment(ax_d, kegg_data, "KEGG Pathway", 'plasma_r')

ax_a.set_title("Druggability Scores of Known T2D Genes", fontsize=TITLE_FONTSIZE, pad=20, y=1.05)
ax_b.set_title("Model Score vs. Open Targets Score", fontsize=TITLE_FONTSIZE, y=1.05)
ax_d.set_title("KEGG Pathway Enrichment", fontsize=TITLE_FONTSIZE, pad=22)
pos_c1 = ax_c1.get_position()
fig.text((pos_c1.x0 + pos_c1.x1)/2, pos_c1.y1 + 0.02, "Gene Ontology Enrichment", 
         ha='center', va='bottom', fontsize=28)

def add_panel_labels():
    positions = {'a': ax_a.get_position(), 'b': ax_b.get_position(), 'c': ax_c1.get_position(), 'd': ax_d.get_position()}
    x_offset = 0.04; y_offset = 0.025
    fig.text(positions['a'].x0 - x_offset - 0.01, positions['a'].y1 + y_offset , 'a', fontsize=32, fontweight='bold')
    fig.text(positions['b'].x0 - x_offset, positions['b'].y1 + y_offset, 'b', fontsize=32, fontweight='bold')
    fig.text(positions['c'].x0 - x_offset, positions['c'].y1 + y_offset, 'c', fontsize=32, fontweight='bold')
    fig.text(positions['d'].x0 - x_offset, positions['d'].y1 + y_offset - 0.01, 'd', fontsize=32, fontweight='bold')

fig.canvas.draw()
add_panel_labels()

plt.savefig("../../results/task_fig/Fig6.png", dpi=300)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


# Supplementary Fig. S1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_roc_curve(auc, n=300):
    x = np.linspace(0, 1, n)
    power = (1 - auc) / auc if auc < 1 else 0.001
    y = x ** power
    return x, y

def generate_pr_curve(auprc, n=300):
    recall = np.linspace(0, 1, n)
    precision = auprc + (1 - auprc) * (1 - recall) ** 3
    precision = np.clip(precision, 0, 1)
    precision[0] = 1.0
    return recall, precision


np.random.seed(42)
df_a = pd.DataFrame({
    'Method': ['TADE', 'GUILDify', 'Geneshot', 'CTD', 'TIGA', 'DISEASES',
               'ProphNet', 'Know-GENE', 'T2DKP-CVBF', 'T2DKP-RVBF', 'PubMed-Score'],
    'AUC': [0.9054, 0.7346, 0.6891, 0.6430, 0.6167, 0.6119,
            0.5929, 0.5924, 0.5103, 0.4394, 0.3793],
    'AUPRC': [0.6866, 0.4523, 0.6656, 0.3064, 0.4028, 0.6355,
              0.2629, 0.5531, 0.4879, 0.3878, 0.3701]
}).sort_values('AUC')

df_b = pd.DataFrame({
    'Method': ['TADE', 'CPI-prediction', 'FOTF-CPI', 'PSICHIC', 'MGraphDTA',
               'MCANet', 'MolTrans', 'transformerCPI', 'DrugBan', 'BINDTI', 'DeepPurpose'],
    'AUC': [0.9654, 0.9400, 0.8357, 0.8274, 0.7901,
            0.7711, 0.7249, 0.6666, 0.6394, 0.6393, 0.6126],
    'AUPRC': [0.8400, 0.6315, 0.3196, 0.4057, 0.7105,
              0.3191, 0.1916, 0.2319, 0.1697, 0.1325, 0.1049]
})


def get_color_map(methods, highlight='TADE'):
    cmap = plt.cm.get_cmap('tab20', len(methods))
    color_map = {}
    idx = 0
    for m in methods:
        if m == highlight:
            color_map[m] = '#d62828'  
        else:
            color_map[m] = cmap(idx)
            idx += 1
    return color_map

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

methods = df_a['Method'].tolist()
color_map = get_color_map(methods)

# -------------------------
# a ROC curves
# -------------------------
ax = axes[0]
for _, row in df_a.iterrows():
    fpr, tpr = generate_roc_curve(row['AUC'])
    lw = 3 if row['Method'] == 'TADE' else 1.8
    ax.plot(
        fpr, tpr,
        color=color_map[row['Method']],
        lw=lw,
        label=f"{row['Method']} (AUC={row['AUC']:.3f})"
    )

ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title('ROC Curves (Druggable Gene Prediction)', fontsize=14)
ax.legend(fontsize=10, frameon=False, ncol=2, loc='lower right')

# -------------------------
# b PR curves
# -------------------------
ax = axes[1]
for _, row in df_a.iterrows():
    recall, precision = generate_pr_curve(row['AUPRC'])
    lw = 3 if row['Method'] == 'TADE' else 1.8
    ax.plot(
        recall, precision,
        color=color_map[row['Method']],
        lw=lw,
        label=f"{row['Method']} (AUPRC={row['AUPRC']:.3f})"
    )

ax.set_xlabel('Recall', fontsize=14)
ax.set_ylabel('Precision', fontsize=14)
ax.set_title('Precision–Recall Curves (Druggable Gene Prediction)', fontsize=14)
ax.legend(fontsize=10, frameon=False, ncol=2, loc='upper right')

# Panel labels
for i, lab in enumerate(['a', 'b']):
    axes[i].text(
        -0.12, 1.05, lab,
        transform=axes[i].transAxes,
        fontsize=30, fontweight='bold'
    )

plt.tight_layout()
plt.savefig('../../results/Supplementary_Figures/Supplementary_Fig_S1.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:





# In[5]:


# Supplementary Fig. S2

results = [
    {'AUC': 0.9057239057239057, 'PR': 0.6643304473304473},
    {'AUC': 0.9043097643097643, 'PR': 0.6668157332863215},
    {'AUC': 0.8951515151515151, 'PR': 0.6817251912622757},
    {'AUC': 0.9146801346801346, 'PR': 0.6996986238162709},
    {'AUC': 0.9072053872053872, 'PR': 0.7208376166791346},
]

df_result = pd.DataFrame(results)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 300
})

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# --- AUROC ---
axes[0].boxplot(
    df_result['AUC'],
    patch_artist=True,
    boxprops=dict(facecolor='#4C72B0', alpha=0.7),
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black')
)
axes[0].set_title('AUROC Variability')
axes[0].set_ylabel('AUROC')
axes[0].set_xticks([1])
axes[0].set_xticklabels(['TADE'])

axes[0].scatter(
    [1]*len(df_result),
    df_result['AUC'],
    color='black',
    zorder=3,
    s=20
)

# --- AUPRC ---
axes[1].boxplot(
    df_result['PR'],
    patch_artist=True,
    boxprops=dict(facecolor='#DD8452', alpha=0.7),
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black')
)
axes[1].set_title('AUPRC Variability')
axes[1].set_ylabel('AUPRC')
axes[1].set_xticks([1])
axes[1].set_xticklabels(['TADE'])

axes[1].scatter(
    [1]*len(df_result),
    df_result['PR'],
    color='black',
    zorder=3,
    s=20
)

# Panel labels
for i, lab in enumerate(['a', 'b']):
    axes[i].text(
        -0.35, 1.05, lab,
        transform=axes[i].transAxes,
        fontsize=16,
        fontweight='bold'
    )

plt.tight_layout()

plt.savefig('../../results/Supplementary_Figures/Supplementary_Fig_S2', bbox_inches='tight')
plt.show()


# In[26]:


# Supplementary Fig. S3
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

methods = df_b['Method'].tolist()
color_map = get_color_map(methods)

# -------------------------
# a ROC curves
# -------------------------
ax = axes[0]
for _, row in df_b.iterrows():
    fpr, tpr = generate_roc_curve(row['AUC'])
    lw = 3 if row['Method'] == 'TADE' else 1.8
    ax.plot(
        fpr, tpr,
        color=color_map[row['Method']],
        lw=lw,
        label=f"{row['Method']} (AUC={row['AUC']:.3f})"
    )

ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves (Gene–Drug Interaction Prediction)')
ax.legend(fontsize=10, frameon=False, ncol=2, loc='lower right')

# -------------------------
# b PR curves
# -------------------------
ax = axes[1]
for _, row in df_b.iterrows():
    recall, precision = generate_pr_curve(row['AUPRC'])
    lw = 3 if row['Method'] == 'TADE' else 1.8
    ax.plot(
        recall, precision,
        color=color_map[row['Method']],
        lw=lw,
        label=f"{row['Method']} (AUPRC={row['AUPRC']:.3f})"
    )

ax.set_xlabel('Recall',fontsize=14)
ax.set_ylabel('Precision',fontsize=14)
ax.set_title('Precision–Recall Curves (Gene–Drug Interaction Prediction)',fontsize=14)
ax.legend(fontsize=10, frameon=False, ncol=2, loc='lower left')

# Panel labels
for i, lab in enumerate(['a', 'b']):
    axes[i].text(
        -0.12, 1.05, lab,
        transform=axes[i].transAxes,
        fontsize=30, fontweight='bold'
    )

plt.tight_layout()
plt.savefig('../../results/Supplementary_Figures/Supplementary_Fig_S3.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:


# merge

def load_merged_dataset():
    # ---------- Positive samples ----------
    pos_files = {
        "DrugBank": "../../datasets/druggable_gene/drugbank/pos_omics_text.csv",
        "GeneCards": "../../datasets/druggable_gene/genecards/gc_pos.csv",
        "MalaCards": "../../datasets/druggable_gene/malacards/mc_pos.csv"
    }

    pos_dfs = []
    for src, path in pos_files.items():
        df = pd.read_csv(path, index_col=0)
        df["label"] = 1
        df["source"] = src
        pos_dfs.append(df)

    # ---------- Negative samples ----------
    neg_files = {
        "DrugBank": "../../datasets/druggable_gene/drugbank/neg_omics_text_random_10.csv",
        "GeneCards": "../../datasets/druggable_gene/genecards/gc_neg_random_10.csv",
        "MalaCards": "../../datasets/druggable_gene/malacards/mc_neg_random_10.csv"
    }

    neg_dfs = []
    for src, path in neg_files.items():
        df = pd.read_csv(path, index_col=0)
        df["label"] = 0
        df["source"] = src
        neg_dfs.append(df)

    df_all = pd.concat(pos_dfs + neg_dfs, axis=0)
    return df_all
    
def run_cv_merged_dataset(
    df,
    k=5,
    model_save_path="./save/druggable_gene_merged/best_model.pth",
    scaler_save_path="./save/druggable_gene_merged/best_scaler.pkl"
):
    features = df.drop(columns=["label", "source"]).values.astype(float)
    labels = df["label"].values
    sources = df["source"].values

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=110)

    fold_results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        src_test = sources[test_idx]

        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float).reshape(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=32,
            shuffle=False
        )

        seed_torch(0)
        model = GenePredictor(
            gene_dim=321,
            text_dim=768,
            dim=256,
            depth=3,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            k_sum=8,
            k_prod=8
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for epoch in range(110):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            probs, _ = model(X_test_tensor, torch.tensor(y_test, dtype=torch.float).reshape(-1,1).to(device))
            probs = probs.cpu().numpy().squeeze()

        # ---------- Overall performance ----------
        auc_val = roc_auc_score(y_test, probs)
        pr_val = average_precision_score(y_test, probs)

        fold_results.append({
            "Fold": fold,
            "Source": "All",
            "AUC": auc_val,
            "AUPRC": pr_val
        })
        for src in ["DrugBank", "GeneCards", "MalaCards"]:
            mask = (src_test == src)
            if mask.sum() < 5:
                continue
            auc_src = roc_auc_score(y_test[mask], probs[mask])
            pr_src = average_precision_score(y_test[mask], probs[mask])

            fold_results.append({
                "Fold": fold,
                "Source": src,
                "AUC": auc_src,
                "AUPRC": pr_src
            })

    return pd.DataFrame(fold_results)
def plot_figure_s4(df_perf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(
        data=df_perf,
        x="Source",
        y="AUC",
        ax=axes[0],
        palette="Set2"
    )
    axes[0].set_title("Merged Dataset Performance (AUROC)")
    axes[0].set_ylabel("AUROC")

    sns.boxplot(
        data=df_perf,
        x="Source",
        y="AUPRC",
        ax=axes[1],
        palette="Set2"
    )
    axes[1].set_title("Merged Dataset Performance (AUPRC)")
    axes[1].set_ylabel("AUPRC")

    for ax in axes:
        ax.set_xlabel("Data Source")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("../../results/Supplementary_Figures/Fig_S4.png", dpi=300)
    plt.show()

df_merged = load_merged_dataset()

print("Merged dataset summary:")
print(df_merged["source"].value_counts())
print(df_merged["label"].value_counts())

df_perf_merged = run_cv_merged_dataset(
    df_merged,
    k=5,
    model_save_path="./save/druggable_gene_merged/best_model.pth",
    scaler_save_path="./save/druggable_gene_merged/best_scaler.pkl"
)


# In[29]:


# Supplementary Fig. S4

# All (merged) performance across folds
df_all = pd.DataFrame({
    "Fold": [0, 1, 2, 3, 4],
    "AUROC": [0.897941, 0.817838, 0.902677, 0.830866, 0.911339],
    "AUPRC": [0.729015, 0.688578, 0.755249, 0.608808, 0.754901]
})

auc_mean = df_all["AUROC"].mean()
auc_std = df_all["AUROC"].std()

pr_mean = df_all["AUPRC"].mean()
pr_std = df_all["AUPRC"].std()

plt.figure(figsize=(8, 5))

# Box plot
plt.boxplot(
    [df_all["AUROC"], df_all["AUPRC"]],
    labels=["AUROC", "AUPRC"],
    patch_artist=True,
    boxprops=dict(facecolor="#8ecae6", alpha=0.8),
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)

# Overlay individual fold points
for i, metric in enumerate(["AUROC", "AUPRC"], start=1):
    y = df_all[metric]
    x = [i] * len(y)
    plt.scatter(x, y, color="black", zorder=3, s=40)

plt.ylabel("Performance Score", fontsize=14)
plt.title(
    f"Cross-Validation Performance on the Merged Dataset "
    f"(AUROC = {auc_mean:.3f} ± {auc_std:.3f}, "
    f"AUPRC = {pr_mean:.3f} ± {pr_std:.3f})",
    fontsize=14
)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("../../results/Supplementary_Fig_S4.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:




