async function get_result(inp) {
  // Simulate a time-consuming task, e.g., fetching data from a server
  console.log("Starting the async function...");
  const settings = {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query: inp }),
  };
  try {
    const fetchResponse = await fetch(
      `http://${"127.0.0.1"}:5000/query`,
      settings
    );
    const data = await fetchResponse.json();
    console.log(data);
    return data.msg;
  } catch (e) {
    console.error(e);
    return e;
  }
}

// let loader = document.getElementById("loader");
// loader.style.display = "none";

document
  .getElementById("submit")
  .addEventListener("click", async function (evt) {
    evt.preventDefault();
    var currentDate = new Date();
    var currentMinute = currentDate.getHours();
    var currentSecond = currentDate.getMinutes();
    let chatBox = document.getElementById("chat-box");
    let inp = document.getElementById("prompt").value;
    document.getElementById("prompt").value = "";
    if (inp == "") return;
    document.getElementById("loader")?.remove();

    chatBox.innerHTML += `<div class="container darker">
    <img src="/static/doctor.jpg" alt="Avatar" class="right" style="width:100%;">
    <p>${inp}</p>
    <span class="time-left">${currentMinute}:${currentSecond}</span>
    </div>`;

    chatBox.innerHTML += `<div
          class="flex w-full items-center justify-center align-center"
          id="loader"
        >
          <center>
            <img src="/static/loading-73.gif" class="w-24 h-24" />
          </center>
        </div>`;

    document.getElementById("loader").style.display = "block";
    chatBox.scrollTo(0, chatBox.scrollHeight);

    chatBox.innerHTML += `<div class="container">
    <img src="/static/q.jpg" alt="Avatar" style="width:100%;">
    <p>${await get_result(inp)}</p>
    <span class="time-right">${currentMinute}:${currentSecond}</span>
    </div>`;

    document.getElementById("loader").style.display = "none";
    chatBox.scrollTo(0, chatBox.scrollHeight);
  });
