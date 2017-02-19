/**
  * function for menu
  * menu contaion mainmenu and submenu
  * if user doesn't click mainmenu, submenu doesn't show
*/
$(document).ready( function() {
  var $submenu = $('.submenu');
  var $mainmenu = $('.mainmenu');

  // if you click a mainmenu item, make the item's submenu to show
  $mainmenu.on('click', 'li', function() {
    $(this).next('.submenu').slideToggle().siblings('.submenu').slideUp();
  });
});