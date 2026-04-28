// Dump prefill logits for a fixed prompt to a JSON file.
// Usage: logit_dump -m model.gguf -p "prompt" -o out.json [-ngl 99]
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "How many positive whole-number divisors does 196 have?";
    std::string outfile = "logits.json";
    int ngl = 99;
    int top_k = 200;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) outfile = argv[++i];
        else if (!strcmp(argv[i], "-ngl") && i + 1 < argc) ngl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) top_k = atoi(argv[++i]);
    }

    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s -m model.gguf [-p prompt] [-o out.json] [-ngl 99] [-k 200]\n", argv[0]);
        return 1;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); llama_model_free(model); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Tokenize
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                  tokens.data(), (int)tokens.size(), true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "Tokenization failed\n");
        llama_free(ctx); llama_model_free(model); return 1;
    }
    tokens.resize(n_tokens);

    fprintf(stderr, "n_tokens=%d\n", n_tokens);
    fprintf(stderr, "tokens:");
    for (int t : tokens) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    // Build batch — output logits for last token only
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        llama_batch_free(batch);
        llama_free(ctx); llama_model_free(model); return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, n_tokens - 1);
    if (!logits) {
        fprintf(stderr, "llama_get_logits_ith returned null\n");
        llama_batch_free(batch);
        llama_free(ctx); llama_model_free(model); return 1;
    }

    // Gather all (token_id, logit) pairs and sort by logit descending
    std::vector<std::pair<float, int>> scored(n_vocab);
    for (int i = 0; i < n_vocab; i++) scored[i] = {logits[i], i};
    std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                      [](const auto & a, const auto & b){ return a.first > b.first; });

    // Write JSON
    std::ofstream f(outfile);
    f << "{\n";
    f << "  \"n_tokens\": " << n_tokens << ",\n";
    f << "  \"tokens\": [";
    for (int i = 0; i < n_tokens; i++) f << tokens[i] << (i + 1 < n_tokens ? "," : "");
    f << "],\n";
    f << "  \"top_logits\": [\n";
    for (int i = 0; i < top_k; i++) {
        f << "    [" << scored[i].second << ", " << scored[i].first << "]";
        if (i + 1 < top_k) f << ",";
        f << "\n";
    }
    f << "  ]\n}\n";
    f.close();

    fprintf(stderr, "Top-3: token=%d logit=%.4f | token=%d logit=%.4f | token=%d logit=%.4f\n",
            scored[0].second, scored[0].first,
            scored[1].second, scored[1].first,
            scored[2].second, scored[2].first);
    fprintf(stderr, "Logits written to %s\n", outfile.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
