
function generate_vis(cluster_data_path,element) {

d3.tsv(cluster_data_path, function loadCallback(error, data) {
  data.forEach(function(d) { // convert strings to numbers
      d.x = +d.x;
      d.y = +d.y;
  });
  makeVis(data,element)});


}


var makeVis = function(data,element) {
// Common pattern for defining vis size and margins
var margin = { top: 0, right: 0, bottom: 0, left: 0 },
    width  = 900 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom; //500

// Add the visualization svg canvas to the vis-container <div>
var canvas = d3.select(element.get(0)).append("svg")
    .attr("width",  width  + margin.left + margin.right)
    .attr("height", height + margin.top  + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Define our scales
var colorScale = d3.scale.category20();

var xScale = d3.scale.linear()
     .domain([-100,100 ])
      .range([0, width]);

    // .domain([ d3.min(data, function(d) { return d.y; }) - 1,
    //           d3.max(data, function(d) { return d.y; }) + 1 ])
   
var yScale = d3.scale.linear()
    .domain([ -100,100]) 
     .range([height, 0]);
    // d3.min(data, function(d) { return d.x; }) - 1,
              // d3.max(data, function(d) { return d.x; }) + 1 ])
    // flip order because y-axis origin is upper LEFT

// Define our axes
var xAxis = d3.svg.axis()
    .scale(xScale)
    .orient('bottom');

var yAxis = d3.svg.axis()
    .scale(yScale)
    .orient('left');

var tooltip = d3.select(element.get(0)).append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

// tooltip mouseover event handler
var tipMouseover = function(d) {
    var color = colorScale(5);
    var html  = d.text + "<br/>" +
                "<span style='color:" + color + ";'>"  + "</span><br/>" +
                "<b> Cluster #" + d.cluster + "</b>";

    tooltip.html(html)
        .style("left", d3.select(this).attr("cx") + "px")     
  .style("top", d3.select(this).attr("cy") + "px")
      .transition()
        .duration(200) // ms
        .style("opacity", .8) // started as 0!

};
// tooltip mouseout event handler
var tipMouseout = function(d) {
    tooltip.transition()
        .duration(300) // ms
        .style("opacity", 0); // don't care about position!
};

// Add data points!
canvas.selectAll(".dot")
  .data(data)
.enter().append("circle")
  .attr("class", "dot")
  .attr("r", 3.5) // radius size, could map to another data dimension
  .attr("cx", function(d) { return xScale( d.y ); })     // x position
  .attr("cy", function(d) { return yScale( d.x ); })  // y position
  .style("fill", function(d) { return colorScale(d.cluster); })
  .on("mouseover", tipMouseover)
  .on("mouseout", tipMouseout);
};



//  ;
//     })
// })(element);


// });