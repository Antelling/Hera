$(function () {
    var values = [];
    $(".best-matches td:nth-child(3)").each(function (i, item) {
        values.push(Number($(item).html()))
    });

    var color = "steelblue";


// A formatter for counts.

    var margin = {top: 0, right: 0, bottom: 0, left: 0},
        width = document.body.scrollWidth - 16 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

    var max = d3.max(values);
    var min = d3.min(values);
    var x = d3.scale.linear()
        .domain([min, max])
        .range([0, width]);

    // Generate a histogram using 50 uniformly-spaced bins.
    var data = d3.layout.histogram()
        .bins(30)
    (values);
    console.log(data);

    var yMax = d3.max(data, function (d) {
        return d.length
    });
    var yMin = d3.min(data, function (d) {
        return d.length
    });
    var colorScale = d3.scale.linear()
        .domain([yMin, yMax])
        .range([d3.rgb(color).brighter(), d3.rgb(color).darker()]);

    var y = d3.scale.linear()
        .domain([0, yMax])
        .range([height, 0]);


    var svg = d3.select("body").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var bar = svg.selectAll(".bar")
        .data(data)
        .enter().append("g")
        .attr("class", "bar")
        .attr("transform", function (d) {
            return "translate(" + x(d.x) + "," + y(d.y) + ")";
        });

    bar.append("rect")
        .attr("x", 1)
        .attr("width", (x(data[0].dx) - x(0)) - 1)
        .attr("height", function (d) {
            return height - y(d.y);
        })
        .attr("fill", function (d) {
            return colorScale(d.y)
        });


    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
});