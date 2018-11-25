
function httpGet(url, callback)
{
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
           // Typical action to be performed when the document is ready:
           data = JSON.parse(xhttp.responseText);
           completeChart(data)
        }
        else {
            console.log("Returned from call.");
        }
    };
    xhttp.open("GET", url, true);
    xhttp.send();
}

/* var data = [4, 8, 15, 16, 23, 42]; */
function drawChart() {
    console.log("Requesting data...")
    httpGet("http://localhost:5000/todo/api/v1.0/data", completeChart);
}

function completeChart(data) {
    console.log("Data: " + data)
    var x = d3.scale.linear()
        .domain([0, d3.max(data)])
        .range([0, 420]);

    d3.select(".chart")
      .selectAll("div")
        .data(data)
      .enter().append("div")
        .style("width", function(d) { return x(d) + "px"; })
        .text(function(d) { return d; });
}
