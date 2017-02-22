
function draw_current_data(convnet, x, y, w, h) {
  push();
  var total_samples = convnet.get_dataset().get_test_size_all();
  var index = convnet.get_dataset().get_sample_index().test;
  var sample_image = convnet.get_test_sample_image();
  var actual = convnet.get_actual_label();
  var predicted = convnet.get_predicted_label();
  var image_length = 80;
  // Whole Frame
  rect(x, y, w, h);
  // Left(image, label)
  image(sample_image, x+20, y+20, image_length, image_length);
  rect(x+20, y+20, image_length, image_length);
  image(labelImage, x+5, y, 48, 48);
  fill(255);
  noStroke();
  textSize(18);
  textStyle(BOLD);
  text(actual, x+20, y+30);
  // Middle(arrow, percentage)
  image(arrowImage, x+130, y+30, 48, 48);
  fill(0);
  textSize(15);
  text(display_text(100.0 * convnet.get_prob()[predicted], 1) + " %", x+130, y+90);
  // Right
  strokeWeight(3);
  stroke(0);
  actual == predicted ? fill(0, 255, 0) : fill(255, 0, 0);
  textSize(52);
  text(predicted, x+210, y+75);
  // Progress
  fill(0);
  noStroke();
  textSize(12);
  textAlign(RIGHT);
  text(index + ' / ' + total_samples, x+w-5, y+h-5);
  pop();
}

function draw_confusion(convnet, x, y, w, h, iy, ix) {
	var results = convnet.get_results();
	var n = convnet.get_dataset().get_classes().length;
	var cell_size = {w:w/n, h:h/n};
  var txtSize = 14;

	var min_precision_wrong = 0.0;
	var max_precision_wrong = 1.0 - results.correct / results.total;
	var min_precision_right = 1.0;
	var max_precision_right = 0.0;

  push();
  textAlign(CENTER);
  translate(x, y);
	for (var a = 0; a < n; a++) {
    for (var p = 0; p < n; p++) {
      var raw = results.confusion[a][p] == 0 ? 0 : results.confusion[a][p];
      var precision = results.predictions[p] == 0 ? 0 : raw / results.predictions[p];
      var recall = results.actuals[a] == 0 ? 0 : raw / results.actuals[a];
      push();
      translate(p * cell_size.w, a * cell_size.h);
      stroke(0, 10);
      if (a == p) {
        var alpha = min_precision_right == max_precision_right ? 255 : map(precision, min_precision_right, max_precision_right, 90, 255);
        fill(0, 255, 0, alpha);
      }
      else {
        var alpha = min_precision_wrong == max_precision_wrong ? 0 : map(precision, min_precision_wrong, max_precision_wrong, 0, 255);
        fill(255, 0, 0, alpha);
      }
      rect(0, 0, cell_size.w, cell_size.h);
      textSize(txtSize);
      fill(0);
      noStroke();
      text(raw, 0.5 * cell_size.w, 0.5 * (cell_size.h + txtSize));
      pop();
    }
  }
  textSize(txtSize);
  strokeWeight(2);
  stroke(0, 150);
  noFill();
  if (ix != -1 && iy != -1) {
    rect(ix * cell_size.w, iy * cell_size.h, cell_size.w, cell_size.h);
  }
  strokeWeight(1);
  for (a = 0; a < n; a++) {
    var pct = results.actuals[a] == 0 ? 0 : results.confusion[a][a] / results.actuals[a];
    push();
    translate((n+0.5) * cell_size.w, a * cell_size.h);
    stroke(0, 150);
    fill(255);
    rect(0, 0, cell_size.w, cell_size.h);
    fill(0);
    noStroke();
    text(results.actuals[a] == 0 ? "-" : floor(100.0 * pct) + "%", 0.5 * cell_size.w, 0.5 * (cell_size.h + txtSize));
    pop();
  }
  for (p = 0; p < n; p++) {
    var pct = results.predictions[p] == 0 ? 0 : results.confusion[p][p] / results.predictions[p];
    push();
    translate(p * cell_size.w, (n+0.5) * cell_size.h);
    stroke(0, 150);
    fill(255);
    rect(0, 0, cell_size.w, cell_size.h);
    fill(0);
    noStroke();
    text(results.predictions[a] == 0 ? "-" : floor(100.0 * pct) + "%", 0.5 * cell_size.w, 0.5 * (cell_size.h + txtSize));
    pop();
  }

  var accuracy = results.total == 0 ? 0 : results.correct / results.total;
  push();
  translate((n+0.5) * cell_size.w, (n+0.5) * cell_size.h);
  stroke(0, 150);
  fill(255);
  rect(-10, -10, cell_size.w+20, cell_size.h+20);
  fill(0);
  noStroke();
  textStyle(BOLD);
  text("Accuracy", 0.5 * cell_size.w, txtSize-6);
  text(display_text(100.0*accuracy, 1)+"%", 0.5 * cell_size.w, 0.5 * (cell_size.h + txtSize));
  pop();

  noStroke(0);
  fill(0);
  textSize(txtSize);
  textAlign(RIGHT);
  for (a = 0; a < n; a++) {
  	push();
    translate(-12, (a + 0.5) * cell_size.h);
    textStyle(a == iy ? BOLD : NORMAL);
    text(a, 0, txtSize/2);
    pop();
	}
  textAlign(LEFT);
  for (p = 0; p < n; p++) {
    push();
    translate((p + 0.5) * cell_size.w, -4);
    textStyle(p == ix ? BOLD : NORMAL);
    text(p, 0, 0);
    pop();
  }
  textStyle(BOLD);
  push();
  textAlign(RIGHT);
  translate(-15, (n + 0.5) * cell_size.h);
  rotate(-PI/2.0);
  text("Precision", 0, txtSize/2);
  text("Actual", 270, -15);
  pop();
  push();
  textAlign(LEFT);
  translate((n + 0.5) * cell_size.w, -10);
  text("Recall", 5, txtSize/2);
  pop();
  push();
  translate(0, 0);
  text("Predicted", 230, -25);
  pop();
  pop();
  stroke(0);
  noFill();
  rect(x, y, w, h);
}

function draw_confusion_samples(convnet, x, y, w, h, actual, predicted) {
	var results = convnet.get_results();
  var max_samples = 28;
  var sample_w = 64;
  var n = results.tops[actual][predicted].length;
  push();
  stroke(0);
  actual == predicted ? fill(0, 255, 0, 50) : fill(255, 0, 0, 50);
  rect(x, y, w, h);
  translate(x, y);
  fill(0);
  noStroke();
  textStyle(BOLD);
  textSize(16);
  text(actual + (actual==predicted ? " classified as " : " misclassified as ") +predicted, 5, 18);
  textStyle(NORMAL);
  for (var i=0; i<min(max_samples, n); i++) {
    var idx_pred_r = floor(i / 4);
    var idx_pred_c = i % 4;
  	var idx = results.tops[actual][predicted][i].idx;
  	var img = convnet.get_test_sample_image(idx);
		var pct = floor(100.0*results.tops[actual][predicted][i].prob);
		push();
  	translate(4 + idx_pred_c * (sample_w+4), 24 + idx_pred_r * (sample_w+4+14));
  	image(img, 2, 2, sample_w, sample_w);
  	stroke(0);
    noFill();
  	rect(2, 2, sample_w, sample_w);
  	noStroke(0);
    fill(0);
    textSize(12);
		text(pct+"%", 2 + 0.5*sample_w, sample_w+14);
  	pop();
  }
  pop();
}

function display_text(n, d) {
  var t = nfs(n, 0, d);
  return t[0] == ' ' ? t.slice(1) : t;
}