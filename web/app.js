const testIdInput = document.querySelector("#testId");
const topNSelect = document.querySelector("#topN");
const recommendBtn = document.querySelector("#recommendBtn");
const randomBtn = document.querySelector("#randomBtn");
const message = document.querySelector("#message");
const sampleTitle = document.querySelector("#sampleTitle");
const sampleMeta = document.querySelector("#sampleMeta");
const userTag = document.querySelector("#userTag");
const recommendations = document.querySelector("#recommendations");
const metricAuc = document.querySelector("#metricAuc");
const metricMovies = document.querySelector("#metricMovies");
const metricSamples = document.querySelector("#metricSamples");

function showMessage(text) {
  message.textContent = text;
  message.hidden = !text;
}

function setLoading(isLoading) {
  recommendBtn.disabled = isLoading;
  randomBtn.disabled = isLoading;
  recommendBtn.textContent = isLoading ? "推理中..." : "生成推荐";
}

function formatScore(score) {
  return Number(score).toFixed(6);
}

function renderResult(data) {
  sampleTitle.textContent = data.sample_movie.title;
  sampleMeta.textContent = `测试 ID ${data.test_id} / 用户 ${data.user_id} / ${data.sample_movie.genres}`;
  userTag.textContent = `User ${data.user_id}`;
  recommendations.innerHTML = data.recommendations.map((item) => `
    <div class="movie-row">
      <div class="rank">${item.rank}</div>
      <div>
        <p class="movie-title">${item.title}</p>
        <p class="movie-genres">${item.genres}</p>
      </div>
      <div class="score">${formatScore(item.score)}</div>
    </div>
  `).join("");
}

async function recommend() {
  const testId = Number(testIdInput.value);
  const topN = Number(topNSelect.value);
  if (!Number.isInteger(testId) || testId < 0) {
    showMessage("请输入有效的非负测试 ID。");
    return;
  }
  setLoading(true);
  showMessage("");
  try {
    const res = await fetch(`/api/recommend?test_id=${testId}&top_n=${topN}`);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "推荐请求失败");
    }
    renderResult(data);
  } catch (err) {
    showMessage(err.message);
  } finally {
    setLoading(false);
  }
}

async function randomSample() {
  const res = await fetch("/api/random");
  const data = await res.json();
  testIdInput.value = data.test_id;
  recommend();
}

async function loadStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  metricAuc.textContent = data.best_auc ? data.best_auc.toFixed(4) : "--";
  metricMovies.textContent = data.num_movies;
  metricSamples.textContent = data.num_samples;
  testIdInput.max = Math.max(0, data.num_samples - 1);
}

recommendBtn.addEventListener("click", recommend);
randomBtn.addEventListener("click", randomSample);
testIdInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    recommend();
  }
});

loadStatus().then(recommend).catch((err) => showMessage(err.message));
