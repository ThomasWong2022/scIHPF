# Running simulations on CMPH by multiprocessing

import numpy as np
import gzip
import os.path
from scipy.sparse import csr_matrix, csc_matrix
import subprocess
import os
import sys
import pickle
from multiprocessing import Pool

import MF

# Read sparse matrix
# QC, Normalisation

MIN_TRANSCRIPTS = 600


def load_tab(fname, max_genes=70000, additionalgenename=False):
    if fname.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open

    with opener(fname, "r") as f:
        if fname.endswith(".gz"):
            header = f.readline().decode("utf-8").rstrip().split("\t")
        else:
            header = f.readline().rstrip().split("\t")

        cells = header[1:]
        X = np.zeros((len(cells), max_genes))
        X2 = np.zeros((len(cells) - 1, max_genes))
        genes = []
        for i, line in enumerate(f):
            if i >= max_genes:
                break
            if fname.endswith(".gz"):
                line = line.decode("utf-8")
            fields = line.rstrip().split("\t")
            if additionalgenename:
                genes.append(fields[1])
                X2[:, i] = [float(f) for f in fields[2:]]
            else:
                genes.append(fields[0])
                X[:, i] = [float(f) for f in fields[1:]]
    if additionalgenename:
        return X2[:, range(len(genes))], np.array(cells), np.array(genes)
    else:
        return X[:, range(len(genes))], np.array(cells), np.array(genes)


def load_mtx(dname):
    with open(dname + "/matrix.mtx", "r") as f:
        while True:
            header = f.readline()
            if not header.startswith("%"):
                break
        header = header.rstrip().split()
        n_genes, n_cells = int(header[0]), int(header[1])

        data, i, j = [], [], []
        for line in f:
            fields = line.rstrip().split()
            data.append(float(fields[2]))
            i.append(int(fields[1]) - 1)
            j.append(int(fields[0]) - 1)
        X = csr_matrix((data, (i, j)), shape=(n_cells, n_genes))

    genes = []
    with open(dname + "/genes.tsv", "r") as f:
        for line in f:
            fields = line.rstrip().split()
            genes.append(fields[1])
    assert len(genes) == n_genes

    return X, np.array(genes)


def process_tab(fname, additionalgenename=False, min_trans=MIN_TRANSCRIPTS):
    X, cells, genes = load_tab(fname, additionalgenename=additionalgenename)
    gt_idx = [i for i, s in enumerate(np.sum(X != 0, axis=1)) if s >= min_trans]
    X = X[gt_idx, :]
    cells = cells[gt_idx]
    if len(gt_idx) == 0:
        print("Warning: 0 cells passed QC in {}".format(fname))
    return X, cells, genes


def process_mtx(dname, min_trans=MIN_TRANSCRIPTS):
    X, genes = load_mtx(dname)
    gt_idx = [i for i, s in enumerate(np.sum(X != 0, axis=1)) if s >= min_trans]
    X = X[gt_idx, :]
    if len(gt_idx) == 0:
        print("Warning: 0 cells passed QC in {}".format(dname))
    return X, genes


##################################################################################################
# Dataset Merging
########################################################################################


def read_real_data(counter):

    # Assume real datasets have cell as rows and genes as columns No index
    # keep track of real data mappings
    datamapping = {
        1: "pancreas_inDrop",
        2: "pancreas_multi_celseq2_expression_matrix",
        3: "pancreas_multi_celseq_expression_matrix",
        4: "pancreas_multi_fluidigmc1_expression_matrix",
        5: "pancreas_multi_smartseq2_expression_matrix",
        6: "293t",
        7: "jurkat",
        8: "jurkat_293t_50_50",
        9: "jurkat_293t_99_1",
        10: "68k_pbmc",
        11: "b_cells",
        12: "cd14_monocytes",
        13: "cd4_t_helper",
        14: "cd56_nk",
        15: "cytotoxic_t",
        16: "memory_t",
        17: "regulatory_t",
        18: "pbmc_10X",
        19: "pbmc_kang",
        20: "nuclei",
        21: "Cerebellum_ALT",
        22: "Cortex_noRep5_FRONTALonly",
        23: "Cortex_noRep5_POSTERIORonly",
        24: "EntoPeduncular",
        25: "GlobusPallidus",
        26: "Hippocampus",
        27: "Striatum",
        28: "SubstantiaNigra",
        29: "Thalamus",
        30: "GSM3589406_PP001swap.filtered.matrix",
        31: "GSM3589407_PP002swap.filtered.matrix",
        32: "GSM3589408_PP003swap.filtered.matrix",
        33: "GSM3589409_PP004swap.filtered.matrix",
    }

    datasetname = datamapping[counter]

    if counter in [1, 2, 3, 4, 5]:
        genecountdata, cell_array, gene_array = process_tab(
            "./Real/scanorama/pancreas/{}.txt.gz".format(datasetname)
        )
        genecountdata = csr_matrix(genecountdata)
        print("Read real dataset {}".format(datasetname))

    if counter in [6, 7, 8, 9]:
        genecountdata, gene_array = process_mtx(
            "./Real/scanorama/293t_jurkat/{}".format(datasetname)
        )
        genecountdata = csr_matrix(genecountdata)
        print("Read real dataset {}".format(datasetname))

    if counter in [10, 11, 12, 13, 14, 15, 16, 17]:
        genecountdata, gene_array = process_mtx(
            "./Real/scanorama/pbmc/10x/{}".format(datasetname)
        )
        genecountdata = csr_matrix(genecountdata)
        print("Read real dataset {}".format(datasetname))

    if counter in [18, 19]:
        genecountdata, cell_array, gene_array = process_tab(
            "./Real/scanorama/pbmc/{}.txt.gz".format(datasetname)
        )
        genecountdata = csr_matrix(genecountdata)
        print("Read real dataset {}".format(datasetname))

    if counter in [20]:
        genecountdata, gene_array = process_mtx(
            "./Real/scanorama/mouse_brain/{}".format(datasetname)
        )
        genecountdata = csr_matrix(genecountdata)
        print("Read real dataset {}".format(datasetname))

    if counter in [21, 22, 23, 24, 25, 26, 27, 28, 29]:
        genecountdata, gene_array = process_mtx(
            "./Real/scanorama/mouse_brain/dropviz/{}".format(datasetname)
        )
        genecountdata = csr_matrix(genecountdata)
        print("Read real dataset {}".format(datasetname))

    if counter in [30, 31, 32, 33, 34]:
        genecountdata, cell_array, gene_array = process_tab(
            "./Real/Levitin_bloodT/{}.txt.gz".format(datasetname), additionalgenename=True
        )
        genecountdata = csr_matrix(genecountdata)
        print("Read real dataset {}".format(datasetname))

    return datasetname, genecountdata, gene_array


def read_cell_labels(counter, mergedcounts):

    # Default cell labels to be batch labels if not given
    dataset_labels = []
    no_datasets = len(mergedcounts)
    for i in range(0, no_datasets):
        datasetsize = mergedcounts[i].shape[0]
        for j in range(0, datasetsize):
            dataset_labels.append(i)

    cell_labels = dataset_labels

    from sklearn.preprocessing import LabelEncoder

    if counter == 1:
        cell_labels = (
            open("./Real/scanorama/cell_labels/pancreas_cluster.txt")
            .read()
            .rstrip()
            .split()
        )

    if counter == 2:
        cell_labels = (
            open("./Real/scanorama/cell_labels/293t_jurkat_cluster.txt")
            .read()
            .rstrip()
            .split()
        )

    if counter == 3:
        cell_labels = (
            open("./Real/scanorama/cell_labels/pbmc_cluster.txt").read().rstrip().split()
        )

    if counter == 4:
        cell_labels = (
            open("./Real/scanorama/cell_labels/newpbmc_cluster.txt")
            .read()
            .rstrip()
            .split()
        )

    if counter == 5:
        cell_labels = (
            open("./Real/scanorama/cell_labels/10xpbmc_cluster.txt")
            .read()
            .rstrip()
            .split()
        )

    if counter <= 5:
        le = LabelEncoder().fit(cell_labels)
        cell_labels = le.transform(cell_labels)
        cell_types = le.classes_
        print("There are {} types of cells".format(len(cell_types)))
        print(cell_types)

    return cell_labels


# Feature selection


def hvg_selection(genecountdata, genes, topgenes):
    # Work on sparse matrix
    threshold = min(np.round(genecountdata.shape[0] * 0.01), 10)
    goodgenes = np.sum(genecountdata > 0, axis=0) > threshold
    filteredcountdata = genecountdata.tocsc()[:, np.where(goodgenes)[1]]
    gene_mean = np.mean(filteredcountdata, axis=0)
    squared_count = filteredcountdata.power(2)
    gene_variances = np.sqrt(squared_count.mean(axis=0) - np.square(gene_mean))
    gene_cv = np.divide(gene_variances, gene_mean)
    gene_rank = np.argsort(gene_cv)[::-1][:, :topgenes]
    selectedcounts = filteredcountdata.tocsc()[:, np.ravel(gene_rank)]
    selected_genes = genes[np.ravel(gene_rank)]
    print(selectedcounts.shape)
    return selectedcounts.tocsr(), selected_genes


# Put datasets into a single matrix with the intersection of all genes.
def merge_datasets(datasets, genes, ds_names=None, verbose=True, union=False):
    if union:
        sys.stderr.write(
            "WARNING: Integrating based on the union of genes is "
            "highly discouraged, consider taking the intersection "
            "or requantifying gene expression.\n"
        )

    # Find genes in common.
    keep_genes = set()
    for idx, gene_list in enumerate(genes):
        if len(keep_genes) == 0:
            keep_genes = set(gene_list)
        elif union:
            keep_genes |= set(gene_list)
        else:
            keep_genes &= set(gene_list)
        if not union and not ds_names is None and verbose:
            print("After {}: {} genes".format(ds_names[idx], len(keep_genes)))
        if len(keep_genes) == 0:
            print("Error: No genes found in all datasets, exiting...")
            exit(1)
    if verbose:
        print("Found {} genes among all datasets".format(len(keep_genes)))

    if union:
        union_genes = sorted(keep_genes)
        for i in range(len(datasets)):
            if verbose:
                print("Processing data set {}".format(i))
            X_new = np.zeros((datasets[i].shape[0], len(union_genes)))
            X_old = csc_matrix(datasets[i])
            gene_to_idx = {gene: idx for idx, gene in enumerate(genes[i])}
            for j, gene in enumerate(union_genes):
                if gene in gene_to_idx:
                    X_new[:, j] = X_old[:, gene_to_idx[gene]].toarray().flatten()
            datasets[i] = csr_matrix(X_new)
        ret_genes = np.array(union_genes)
    else:
        # Only keep genes in common.
        ret_genes = np.array(sorted(keep_genes))
        for i in range(len(datasets)):
            # Remove duplicate genes.
            uniq_genes, uniq_idx = np.unique(genes[i], return_index=True)
            datasets[i] = datasets[i].tocsc()[:, uniq_idx]
            # Do gene filtering.
            gene_sort_idx = np.argsort(uniq_genes)
            gene_idx = [idx for idx in gene_sort_idx if uniq_genes[idx] in keep_genes]
            datasets[i] = datasets[i][:, gene_idx].tocsr()
            assert np.array_equal(uniq_genes[gene_idx], ret_genes)

    return datasets, ret_genes


# Running MF methods


def MF_run(simulate_method, task, counter, outputdir, topgenes=40000, a=0.3, c=0.3):

    # Sparsim output genes as rows and cells as columns s
    if simulate_method != "real":

        print("Not implemented")

    else:
        # Read and process datasets

        # Data Integration tasks
        if task in ["integration", "integration_hyper", "integration_geneselection"]:
            # Mapping the list of datasets to use
            optionmapping = {
                1: [1, 2, 3, 4, 5],
                2: [6, 7, 8],
                3: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                4: [10, 18, 19],
                5: [
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                ],
            }
            optimal_n_factors = {
                1: 10,
                2: 2,
                3: 7,
                4: 5,
                5: 7,
            }
            optimal_noise = {
                1: 0.1,
                2: 0.1,
                3: 0.1,
                4: 0.1,
                5: 0.1,
            }
            # Read a list of numpy array or
            datasetnames = []
            genecountdatalist = []
            genelist = []
            for c in optionmapping[counter]:
                datasetname, genecountdata, genes = read_real_data(c)
                # sparse = np.sum(np.sum(genecountdata>0))/genecountdata.size
                # print('Sparsity {}'.format(sparse))
                processedcountdata, filtered_genes = hvg_selection(
                    genecountdata, genes, topgenes
                )
                datasetnames.append(datasetname)
                genecountdatalist.append(processedcountdata)
                genelist.append(filtered_genes)

            mergedgenecountdata, shared_genes = merge_datasets(
                genecountdatalist, genelist
            )

            # Read cell labels
            cell_labels = read_cell_labels(counter, mergedgenecountdata)

            # optimal n factors for hyper
            startK = optimal_n_factors[counter]
            endK = optimal_n_factors[counter]
            noise_ratio = optimal_noise[counter]

        # Read a single dataset
        else:
            datasetname, genecountdata, genelist = read_real_data(counter)
            # sparse = np.sum(np.sum(genecountdata>0))/genecountdata.size
            # print('Sparsity {}'.format(sparse))
            processedcountdata, filtered_genes = hvg_selection(
                genecountdata, genelist, topgenes
            )

        # Perform various tasks on scanorama datasets
        if task == "integration":
            MF_results, MF_variance, MF_clustering = MF.MF_integration(
                mergedgenecountdata, shared_genes, cell_labels
            )
            filename = "./Result/{}_integration/Dataset_{}_top{}_ks.json".format(
                outputdir, outputdir, topgenes
            )
            import json

            with open(filename, "w") as fp:
                json.dump(MF_variance, fp)
            filename = "./Result/{}_integration/Dataset_{}_top{}_scores".format(
                outputdir, outputdir, topgenes
            )
            outfile = open(filename, "wb")
            pickle.dump(MF_results, outfile)
            outfile.close()
            filename = "./Result/{}_integration/Dataset_{}_top{}_clustering".format(
                outputdir, outputdir, topgenes
            )
            outfile = open(filename, "wb")
            pickle.dump(MF_clustering, outfile)
            outfile.close()

        if task == "integration_hyper":
            MF_results, MF_variance, MF_clustering = MF.MF_integration_hyper(
                mergedgenecountdata, shared_genes, cell_labels, startK=startK, endK=endK
            )
            filename = "./Result/{}_{}/Dataset_{}_top{}_ks.json".format(
                outputdir, task, outputdir, topgenes
            )
            import json

            with open(filename, "w") as fp:
                json.dump(MF_variance, fp)
            filename = "./Result/{}_{}/Dataset_{}_top{}_scores".format(
                outputdir, task, outputdir, topgenes
            )
            outfile = open(filename, "wb")
            pickle.dump(MF_results, outfile)
            outfile.close()
            filename = "./Result/{}_{}/Dataset_{}_top{}_clustering".format(
                outputdir, task, outputdir, topgenes
            )
            outfile = open(filename, "wb")
            pickle.dump(MF_clustering, outfile)
            outfile.close()

        if task == "integration_geneselection":
            MF_results, MF_variance, MF_clustering = MF.MF_integration(
                mergedgenecountdata, shared_genes, cell_labels
            )
            filename = "./Result/{}_{}/Dataset_{}_top{}_ks.json".format(
                outputdir, task, outputdir, topgenes, noise=noise_ratio
            )
            import json

            with open(filename, "w") as fp:
                json.dump(MF_variance, fp)
            filename = "./Result/{}_{}/Dataset_{}_top{}_scores".format(
                outputdir, task, outputdir, topgenes
            )
            outfile = open(filename, "wb")
            pickle.dump(MF_results, outfile)
            outfile.close()
            filename = "./Result/{}_{}/Dataset_{}_top{}_clustering".format(
                outputdir, task, outputdir, topgenes
            )
            outfile = open(filename, "wb")
            pickle.dump(MF_clustering, outfile)
            outfile.close()

        if task == "factor":
            MF_results, MF_variance, MF_clustering = MF.MF_factors(
                processedcountdata, topgenes=topgenes
            )
            filename = "./Result/{}_factor/Dataset_{}_top{}_ks.json".format(
                outputdir, datasetname, topgenes
            )
            import json

            with open(filename, "w") as fp:
                json.dump(MF_variance, fp)
            filename = "./Result/{}_factor/Dataset_{}_top{}_scores".format(
                outputdir, datasetname, topgenes
            )
            outfile = open(filename, "wb")
            pickle.dump(MF_results, outfile)
            outfile.close()
            filename = "./Result/{}_factor/Dataset_{}_top{}_clustering".format(
                outputdir, datasetname, topgenes
            )
            outfile = open(filename, "wb")
            pickle.dump(MF_clustering, outfile)
            outfile.close()

        if task == "benchmark":
            MF_results, MF_variance = MF.MF_genes(processedcountdata, topgenes=topgenes)
            filename = "./Result/{}_benchmark/Dataset_{}_top{}_variance.json".format(
                outputdir, datasetname, topgenes
            )
            import json

            with open(filename, "w") as fp:
                json.dump(MF_variance, fp)
            filename = "./Result/{}_benchmark/Dataset_{}_top{}_scores".format(
                outputdir, datasetname, topgenes
            )
            outfile = open(filename, "wb")
            pickle.dump(MF_results, outfile)
            outfile.close()

        if task == "hyper":
            MF_variance = MF.MF_hyperparameter(processedcountdata, a=a, c=c)
            filename = "./Result/{}_hyper/Dataset_{}_hyper_{}_{}.json".format(
                outputdir, datasetname, a, c
            )
            import json

            with open(filename, "w") as fp:
                json.dump(MF_variance, fp)


if __name__ == "__main__":

    method = sys.argv[1]
    task = sys.argv[2]
    start = int(sys.argv[3])

    np.random.seed(0)

    datamapping = {
        1: "pancreas_inDrop",
        2: "pancreas_multi_celseq2_expression_matrix",
        3: "pancreas_multi_celseq_expression_matrix",
        4: "pancreas_multi_fluidigmc1_expression_matrix",
        5: "pancreas_multi_smartseq2_expression_matrix",
        6: "293t",
        7: "jurkat",
        8: "jurkat_293t_50_50",
        9: "jurkat_293t_99_1",
        10: "68k_pbmc",
        11: "b_cells",
        12: "cd14_monocytes",
        13: "cd4_t_helper",
        14: "cd56_nk",
        15: "cytotoxic_t",
        16: "memory_t",
        17: "regulatory_t",
        18: "pbmc_10X",
        19: "pbmc_kang",
        20: "nuclei",
        21: "Cerebellum_ALT",
        22: "Cortex_noRep5_FRONTALonly",
        23: "Cortex_noRep5_POSTERIORonly",
        24: "EntoPeduncular",
        25: "GlobusPallidus",
        26: "Hippocampus",
        27: "Striatum",
        28: "SubstantiaNigra",
        29: "Thalamus",
    }

    integrationmapping = {
        1: "humanpancreas",
        2: "10Xmouse",
        3: "10Xpbmc",
        4: "newpbmc",
        5: "hpfpbmc",
    }

    outputdir = datamapping[start]
    if task in ["integration", "integration_hyper", "integration_geneselection"]:
        outputdir = integrationmapping[start]

    if not os.path.exists("./Result/{}_{}".format(outputdir, task)):
        os.mkdir("./Result/{}_{}".format(outputdir, task))

    topgenelist = [
        2500,
        5000,
        10000,
        25000,
        50000,
    ]

    if task == "hyper":
        with Pool() as pool:
            for r in range(start, start + 1):
                for a in np.linspace(0.1, 2.1, num=11):
                    for c in np.linspace(0.1, 2.1, num=11):
                        print("Grid Search {} {}".format(a, c))
                        MF_run(method, task, r, outputdir, a=a, c=c)
                        # res = pool.apply_async(func=Sparsim_run, args=(method,r,outputdir,a,ap))
            pool.close()
            pool.join()

    elif task in [
        "integration",
    ]:
        with Pool() as pool:
            for r in range(start, start + 1):
                res = pool.apply_async(
                    func=MF_run, args=(method, task, r, outputdir, 40000)
                )
            pool.close()
            pool.join()

    else:
        with Pool() as pool:
            for r in range(start, start + 1):
                for topgenes in topgenelist:
                    print("Top genes selection {} {}".format(topgenes, outputdir))
                    res = pool.apply_async(
                        func=MF_run, args=(method, task, r, outputdir, topgenes)
                    )
            pool.close()
            pool.join()
